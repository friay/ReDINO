import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, repo_dir, weights, use_bf16=True,):
        super().__init__()

        self.backbone = torch.hub.load(
            repo_or_dir=repo_dir,
            model="dinov3_vitb16",
            source="local",
            weights=weights,
            pretrained=True,
        )

        # Full fine-tuning: all original DINOv3 parameters are trainable.
        self.backbone.requires_grad_(True)

        self.interaction_indexes = [2, 5, 8, 11]
        self.use_bf16 = use_bf16
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),)
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        x = (x - self.img_mean) / self.img_std

        if self.use_bf16 and x.is_cuda:
            amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16,)
        else:
            amp_context = contextlib.nullcontext()

        with amp_context:
            all_layers = self.backbone.get_intermediate_layers(
                x,
                n=self.interaction_indexes,
                reshape=True,
                norm=True,
            )

        x1, x2, x3, x4 = all_layers
        H, W = x1.shape[-2:]

        # Same feature pyramid construction as Frozen DINOv3 baseline.
        x1 = F.interpolate(x1, size=(4 * H, 4 * W), mode="bilinear", align_corners=False,)
        x2 = F.interpolate(x2, size=(2 * H, 2 * W), mode="bilinear", align_corners=False,)
        # x3 stays at native DINOv3 patch resolution.
        x4 = F.interpolate(x4, size=(max(1, H // 2), max(1, W // 2)), mode="bilinear", align_corners=False,)

        return x1, x2, x3, x4


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False,)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False,),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False,),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False,)
            x = torch.cat([x, skip], dim=1)

        x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_classes, encoder_dim=768):
        super().__init__()

        self.up1 = DecoderBlock(in_channels=encoder_dim, skip_channels=encoder_dim, out_channels=512,)
        self.up2 = DecoderBlock(in_channels=512, skip_channels=encoder_dim, out_channels=256,)
        self.up3 = DecoderBlock(in_channels=256, skip_channels=encoder_dim, out_channels=128,)
        self.up4 = DecoderBlock(in_channels=128, skip_channels=0, out_channels=64,)
        self.up5 = DecoderBlock(in_channels=64, skip_channels=0, out_channels=32,)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1,)

    def forward(self, features):
        f1, f2, f3, f4 = features

        x = self.up1(f4, f3)
        x = self.up2(x, f2)
        x = self.up3(x, f1)
        x = self.up4(x, None)
        x = self.up5(x, None)

        return self.final_conv(x)


class FullFineTuneDINOv3Seg(nn.Module):
    def __init__(
        self,
        num_classes,
        repo_dir,
        weights,
        use_bf16=True,
    ):
        super().__init__()

        self.encoder = Encoder(
            repo_dir=repo_dir,
            weights=weights,
            use_bf16=use_bf16,
        )

        self.decoder = Decoder(
            num_classes=num_classes,
            encoder_dim=self.encoder.backbone.embed_dim,
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))



if __name__ == "__main__":
    model = FullFineTuneDINOv3Seg(num_classes=9, use_bf16=False,)

    x = torch.randn(1, 3, 224, 224)
    output = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dino_params = sum(p.numel() for p in model.encoder.backbone.parameters())
    trainable_dino_params = sum(p.numel() for p in model.encoder.backbone.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in model.decoder.parameters())

    print("-" * 60)
    print(f"Total params            : {total_params:,}")
    print(f"Total trainable params  : {trainable_params:,}")
    print(f"DINOv3 params           : {dino_params:,}")
    print(f"Trainable DINOv3 params : {trainable_dino_params:,}")
    print(f"Decoder params          : {decoder_params:,}")
