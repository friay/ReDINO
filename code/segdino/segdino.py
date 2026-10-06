import torch
import torch.nn as nn
import torch.nn.functional as F


class SegDINOHead(nn.Module):
    """
    SegDINO V1 lightweight decoder.
    """
    def __init__(
        self,
        num_classes,
        in_channels=768,
        features=128,
        out_channels=(96, 192, 384, 768),
    ):
        super().__init__()

        # Project four DINO intermediate features
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels,
                out_channel,
                kernel_size=1,
                stride=1,
                padding=0
            )
            for out_channel in out_channels
        ])

        # Unify channel dimensions
        self.layer1_rn = nn.Conv2d(out_channels[0], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(out_channels[1], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(out_channels[2], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(out_channels[3], features, kernel_size=3, stride=1, padding=1, bias=False)

        # Upsample the first intermediate feature by 4x
        self.proj = nn.ConvTranspose2d(
            features,
            features,
            kernel_size=4,
            stride=4,
            padding=0,
            bias=False
        )

        # Fuse four features
        self.output_conv = nn.Conv2d(
            features * 4,
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, features, patch_h, patch_w):
        out = []
        # Token -> feature map -> channel projection
        for i, x in enumerate(features):
            B, N, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, patch_h, patch_w)
            x = self.projects[i](x)
            out.append(x)

        layer1, layer2, layer3, layer4 = out
        layer1 = self.layer1_rn(layer1)
        layer2 = self.layer2_rn(layer2)
        layer3 = self.layer3_rn(layer3)
        layer4 = self.layer4_rn(layer4)

        layer1 = self.proj(layer1)
        target_size = layer1.shape[-2:]

        layer2 = F.interpolate(layer2, size=target_size, mode="bilinear", align_corners=True)
        layer3 = F.interpolate(layer3, size=target_size, mode="bilinear", align_corners=True)
        layer4 = F.interpolate(layer4, size=target_size, mode="bilinear", align_corners=True)

        # Feature fusion
        fused = torch.cat([layer1, layer2, layer3, layer4], dim=1)
        logits = self.output_conv(fused)

        return logits


class SegDINO(nn.Module):

    def __init__(
        self,
        num_classes,
        repo_dir,
        weights,
    ):
        super().__init__()

        self.backbone = torch.hub.load(
            repo_or_dir=repo_dir,
            model="dinov3_vitb16",
            source="local",
            weights=weights,
            pretrained=True,
        )

        self.backbone.requires_grad_(False)
        self.backbone.eval()

        # Same layers used by SegDINO V1
        self.intermediate_layers = [2, 5, 8, 11]

        # SegDINO lightweight decoder
        self.head = SegDINOHead(
            num_classes=num_classes,
            in_channels=self.backbone.embed_dim,  # 768 for ViT-B
            features=128,
            out_channels=(96, 192, 384, 768),
        )

        # ImageNet normalization
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        B, C, H, W = x.shape
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x - self.img_mean) / self.img_std
        patch_h = H // 16
        patch_w = W // 16

        with torch.no_grad():
            features = self.backbone.get_intermediate_layers(x, n=self.intermediate_layers)

        logits = self.head(features, patch_h, patch_w)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=True)
        return logits


if __name__ == "__main__":
    model = SegDINO(num_classes=9)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print("Output:", y.shape)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in model.backbone.parameters())

    print("-" * 50)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Backbone params: {backbone_params:,}")