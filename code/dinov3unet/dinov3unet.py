import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Sequence, Tuple

from models.dinov3_adapter import DINOv3_Adapter


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1,
                      groups=in_channels, bias=bias),
            nn.Conv2d(in_channels, out_channels, 1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LearnableUpsampleBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.up2 = nn.ConvTranspose2d(channels, channels, 2, 2, bias=True)

    def forward(self, x, target_size: Tuple[int, int]):
        h, w = x.shape[-2:]
        out = x
        while h * 2 <= target_size[0] and w * 2 <= target_size[1]:
            out = self.up2(out)
            h, w = out.shape[-2:]
        if (h, w) != target_size:
            out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
        return out


class FAPM(nn.Module):
    """Feature Adaptive Projection Module used by Dino U-Net."""
    def __init__(
        self,
        in_channels: int,
        rank: int = 256,
        out_channels: Sequence[int] = (32, 64, 128, 256),
        bias: bool = False,
    ):
        super().__init__()
        self.out_channels = list(out_channels)

        self.shared_basis = nn.Conv2d(in_channels, rank, 1, bias=bias)
        self.specific_bases = nn.ModuleList([
            nn.Conv2d(in_channels, rank, 1, bias=bias)
            for _ in self.out_channels
        ])
        self.film_generators = nn.ModuleList([
            nn.Conv2d(rank, 2 * rank, 1, bias=bias)
            for _ in self.out_channels
        ])

        self.refinement_blocks = nn.ModuleList()
        self.shortcut_projections = nn.ModuleList()
        for out_ch in self.out_channels:
            self.refinement_blocks.append(nn.Sequential(
                nn.Conv2d(rank, out_ch, 1, bias=bias),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                DepthwiseSeparableConv(out_ch, out_ch, bias=bias),
                nn.Conv2d(out_ch, out_ch, 1, bias=bias),
                SqueezeExcitation(out_ch),
            ))
            self.shortcut_projections.append(
                nn.Conv2d(rank, out_ch, 1, bias=bias) if rank != out_ch else nn.Identity()
            )

    def forward(self, x_list: List[torch.Tensor]):
        if len(x_list) != len(self.out_channels):
            raise ValueError(f"FAPM expects {len(self.out_channels)} features, got {len(x_list)}")

        outputs = []
        for i, x in enumerate(x_list):
            z_shared = self.shared_basis(x)
            z_specific = self.specific_bases[i](x)
            gamma_beta = self.film_generators[i](z_shared)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
            z = gamma * z_specific + beta
            outputs.append(self.refinement_blocks[i](z) + self.shortcut_projections[i](z))
        return outputs


class DinoUNetEncoder(nn.Module):
    def __init__(
        self,
        repo_dir,
        weights,
        target_channels=(32, 64, 128, 256),
        fapm_rank=256,
        normalize_input=True,
        with_cp=False,
    ):
        super().__init__()
        self.target_channels = list(target_channels)
        self.normalize_input = normalize_input

        self.backbone = torch.hub.load(
            repo_or_dir=repo_dir,
            model="dinov3_vitb16",
            source="local",
            weights=weights,
            pretrained=True,
        )
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        self.adapter = DINOv3_Adapter(
            backbone=self.backbone,
            interaction_indexes=[2, 5, 8, 11],
            pretrain_size=512,
            conv_inplane=64,
            n_points=4,
            deform_num_heads=16,
            drop_path_rate=0.3,
            init_values=0.0,
            with_cffn=True,
            cffn_ratio=0.25,
            deform_ratio=0.5,
            add_vit_feature=True,
            use_extra_extractor=True,
            with_cp=with_cp,
        )

        self.fapm = FAPM(
            in_channels=self.backbone.embed_dim,
            rank=fapm_rank,
            out_channels=self.target_channels,
            bias=False,
        )
        self.ups = nn.ModuleList([LearnableUpsampleBlock(ch) for ch in self.target_channels])

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

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        self.adapter.backbone.eval()
        return self

    @staticmethod
    def _adapter_outputs_to_list(feats):
        if isinstance(feats, dict):
            return [feats[str(i)] for i in range(1, 5)]
        if isinstance(feats, (tuple, list)):
            if len(feats) != 4:
                raise ValueError(f"Adapter returned {len(feats)} features; expected 4")
            return list(feats)
        raise TypeError(f"Unsupported adapter output type: {type(feats).__name__}")

    def forward(self, x):
        _, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)
        elif c != 3:
            x = x[:, :3]

        if self.normalize_input:
            x = (x - self.img_mean) / self.img_std

        feats = self._adapter_outputs_to_list(self.adapter(x))
        feats = self.fapm(feats)

        skips = []
        for i, feat in enumerate(feats):
            target_size = (max(1, h // (2 ** i)), max(1, w // (2 ** i)))
            skips.append(self.ups[i](feat, target_size))
        return skips


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2, bias=False):
        super().__init__()
        layers = []
        for i in range(num_convs):
            cin = in_channels if i == 0 else out_channels
            layers.extend([
                nn.Conv2d(cin, out_channels, 3, padding=1, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DinoUNetDecoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        encoder_channels=(32, 64, 128, 256),
        num_convs_per_stage=(2, 2, 2),
        conv_bias=False,
    ):
        super().__init__()
        encoder_channels = list(encoder_channels)
        self.transpconvs = nn.ModuleList()
        self.stages = nn.ModuleList()

        for stage_idx in range(3):
            input_below = encoder_channels[-1 - stage_idx]
            input_skip = encoder_channels[-2 - stage_idx]
            self.transpconvs.append(
                nn.ConvTranspose2d(input_below, input_skip, 2, 2, bias=conv_bias)
            )
            self.stages.append(
                ConvBlock(
                    in_channels=2 * input_skip,
                    out_channels=input_skip,
                    num_convs=num_convs_per_stage[stage_idx],
                    bias=conv_bias,
                )
            )

        self.seg_head = nn.Conv2d(encoder_channels[0], num_classes, 1)

    def forward(self, skips):
        x = skips[-1]
        for stage_idx in range(3):
            x = self.transpconvs[stage_idx](x)
            skip = skips[-2 - stage_idx]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.stages[stage_idx](x)
        return self.seg_head(x)


class DinoUNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        repo_dir,
        weights,
        target_channels=(32, 64, 128, 256),
        fapm_rank=256,
        decoder_convs=(2, 2, 2),
        normalize_input=True,
        adapter_with_cp=False,
    ):
        super().__init__()
        self.encoder = DinoUNetEncoder(
            repo_dir=repo_dir,
            weights=weights,
            target_channels=target_channels,
            fapm_rank=fapm_rank,
            normalize_input=normalize_input,
            with_cp=adapter_with_cp,
        )
        self.decoder = DinoUNetDecoder(
            num_classes=num_classes,
            encoder_channels=target_channels,
            num_convs_per_stage=decoder_convs,
            conv_bias=False,
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


DinoUNet_Synapse = DinoUNet


if __name__ == "__main__":
    model = DinoUNet(num_classes=9, normalize_input=True, adapter_with_cp=False)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in model.encoder.backbone.parameters())
    adapter_params = sum(p.numel() for p in model.encoder.adapter.parameters()) - backbone_params
    fapm_params = sum(p.numel() for p in model.encoder.fapm.parameters())
    alignment_params = sum(p.numel() for p in model.encoder.ups.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())

    print("-" * 60)
    print(f"Total params                : {total_params:,}")
    print(f"Trainable params            : {trainable_params:,}")
    print(f"Frozen DINOv3 backbone      : {backbone_params:,}")
    print(f"Official DINOv3 adapter     : {adapter_params:,}")
    print(f"FAPM                        : {fapm_params:,}")
    print(f"Learnable spatial alignment : {alignment_params:,}")
    print(f"Decoder                     : {decoder_params:,}")
