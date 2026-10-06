import torch
import torch.nn as nn
from models.diff_deform_adapter import DINOv3_Adapter


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load(
                        repo_or_dir="dinov3/",
                        model="dinov3_vitb16",
                        source="local",
                        weights="dinov3_model/",
                        pretrained=True,
                    )

        self.backbone.requires_grad_(False)
        self.backbone.eval()

        self.adapter = DINOv3_Adapter(
            self.backbone,
            interaction_indexes=[2, 5, 8, 11],
            deform_num_heads=12,
            deform_ratio=1,
            n_points=4,
            cffn_ratio=1,
            drop_path_rate=0.1,
            with_cp=False,
        )

        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        self.adapter.backbone.eval()
        return self

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x - self.img_mean) / self.img_std
        return self.adapter(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_classes, encoder_dim=768):
        super().__init__()
        self.up1 = DecoderBlock(in_channels=encoder_dim, skip_channels=encoder_dim, out_channels=512)
        self.up2 = DecoderBlock(in_channels=512, skip_channels=encoder_dim, out_channels=256)
        self.up3 = DecoderBlock(in_channels=256, skip_channels=encoder_dim, out_channels=128)
        self.up4 = DecoderBlock(in_channels=128, skip_channels=0, out_channels=64)
        self.up5 = DecoderBlock(in_channels=64, skip_channels=0, out_channels=32)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, features):
        f1, f2, f3, f4 = features

        x = self.up1(f4, f3)
        x = self.up2(x, f2)
        x = self.up3(x, f1)

        x = self.up4(x, None)
        x = self.up5(x, None)

        logits = self.final_conv(x)
        return logits


class Dino_seg(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_classes)

    def forward(self, x):
        return self.decoder(self.encoder(x))


if __name__ == "__main__":
    model = Dino_seg(num_classes=9)
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    for x in output:
        print(x.shape)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dino_params = sum(p.numel() for p in model.encoder.backbone.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.adapter.parameters()) - dino_params
    decoder_params = sum(p.numel() for p in model.decoder.parameters())

    print("-" * 50)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"encoder可训练参数量: {dino_params:,}")
    print(f"decoder参数量: {decoder_params:,}")

