import torch
import torch.nn as nn


from vit_adapter_dinov3 import ViTAdapterDINOv3


class Encoder(nn.Module):
    def __init__(
        self,
        repo_dir,
        weights,
        normalize_input=True,
        with_cp=False,
        freeze_backbone=False,
    ):
        super().__init__()
        self.normalize_input = normalize_input

        self.backbone = torch.hub.load(
            repo_or_dir=repo_dir,
            model="dinov3_vitb16",
            source="local",
            weights=weights,
            pretrained=True,
        )

        self.freeze_backbone = bool(freeze_backbone)

        # ViT-Adapter-B configuration adapted to DINOv3-B/16.
        self.adapter = ViTAdapterDINOv3(
            backbone=self.backbone,
            interaction_indexes=((0, 2), (3, 5), (6, 8), (9, 11)),
            conv_inplane=64,
            n_points=4,
            deform_num_heads=12,
            drop_path_rate=0.3,
            init_values=0.0,
            with_cffn=True,
            cffn_ratio=0.25,
            deform_ratio=0.5,
            add_vit_feature=True,
            use_extra_extractor=True,
            with_cp=with_cp,
            freeze_backbone=self.freeze_backbone,
        )

        self.register_buffer(
            "img_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "img_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def set_freeze_backbone(self, freeze: bool = True):
        self.freeze_backbone = bool(freeze)
        self.adapter.set_freeze_backbone(self.freeze_backbone)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(
                f"Expected 1-channel or 3-channel input, got C={x.shape[1]}."
            )

        if self.normalize_input:
            x = (x - self.img_mean) / self.img_std

        return self.adapter(x)


class DecoderBlock(nn.Module):
    """Same U-Net-like decoder block used in ReDINO implementation."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, num_classes, encoder_dim=768):
        super().__init__()

        self.up1 = DecoderBlock(in_channels=encoder_dim, skip_channels=encoder_dim, out_channels=512,)
        self.up2 = DecoderBlock(in_channels=512, skip_channels=encoder_dim, out_channels=256,)
        self.up3 = DecoderBlock(in_channels=256, skip_channels=encoder_dim, out_channels=128,)
        self.up4 = DecoderBlock(in_channels=128, skip_channels=0, out_channels=64,)
        self.up5 = DecoderBlock(in_channels=64, skip_channels=0, out_channels=32,)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, features):
        f1, f2, f3, f4 = features

        x = self.up1(f4, f3)
        x = self.up2(x, f2)
        x = self.up3(x, f1)
        x = self.up4(x)
        x = self.up5(x)
        return self.final_conv(x)


class ViTAdapterSeg(nn.Module):
    def __init__(
        self,
        num_classes,
        repo_dir,
        weights,
        normalize_input=True,
        with_cp=False,
        freeze_backbone=False,
    ):
        super().__init__()
        self.freeze_backbone = bool(freeze_backbone)
        self.encoder = Encoder(
            repo_dir=repo_dir,
            weights=weights,
            normalize_input=normalize_input,
            with_cp=with_cp,
            freeze_backbone=self.freeze_backbone,
        )
        self.decoder = Decoder(num_classes=num_classes, encoder_dim=768)

    def set_freeze_backbone(self, freeze: bool = True):
        self.freeze_backbone = bool(freeze)
        self.encoder.set_freeze_backbone(self.freeze_backbone)
        return self

    def forward(self, x):
        return self.decoder(self.encoder(x))



if __name__ == "__main__":
    # Main comparison: closest to original ViT-Adapter training behavior
    model = ViTAdapterSeg(
        num_classes=9,
        with_cp=False,
        freeze_backbone=False,
    )

    # For the controlled parameter-efficiency experiment instead use:
    # model = ViTAdapterSeg(
    #     num_classes=9,
    #     with_cp=False,
    #     freeze_backbone=True,
    # )

    x = torch.randn(1, 3, 224, 224)
    y = model(x)

    print("Input shape              :", x.shape)
    print("Output shape             :", y.shape)
    print("Backbone frozen          :", model.freeze_backbone)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in model.encoder.backbone.parameters())
    backbone_trainable = sum(p.numel() for p in model.encoder.backbone.parameters() if p.requires_grad)
    adapter_total = sum(p.numel() for p in model.encoder.adapter.parameters())
    adapter_only_params = adapter_total - backbone_params
    adapter_only_trainable = sum(
        p.numel()
        for name, p in model.encoder.adapter.named_parameters()
        if not name.startswith("backbone.") and p.requires_grad
    )
    decoder_params = sum(p.numel() for p in model.decoder.parameters())

    print("-" * 64)
    print(f"Total params                 : {total_params:,}")
    print(f"Total trainable params       : {trainable_params:,}")
    print(f"DINOv3 backbone params       : {backbone_params:,}")
    print(f"DINOv3 trainable params      : {backbone_trainable:,}")
    print(f"ViT-Adapter-only params      : {adapter_only_params:,}")
    print(f"ViT-Adapter trainable params : {adapter_only_trainable:,}")
    print(f"Decoder params               : {decoder_params:,}")

    # Optional runtime switch:
    # model.set_freeze_backbone(True)
    # model.set_freeze_backbone(False)
