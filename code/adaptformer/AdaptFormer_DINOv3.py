import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptFormerAdapter(nn.Module):
    """
    AdaptFormer bottleneck branch following the official implementation:

        x -> Linear(d_model, bottleneck) -> ReLU -> Dropout
          -> Linear(bottleneck, d_model) -> scale

    For the standard image configuration used by AdaptFormer:
        bottleneck = 64
        scale = 0.1
        dropout = 0.1
        adapter LayerNorm = none
        init = "lora" style:
            down_proj: Kaiming uniform
            up_proj:   zeros
            biases:    zeros
    """

    def __init__(
        self,
        d_model=768,
        bottleneck=64,
        dropout=0.1,
        adapter_scalar=0.1,
        learnable_scalar=False,
    ):
        super().__init__()

        self.down_proj = nn.Linear(d_model, bottleneck)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(bottleneck, d_model)

        if learnable_scalar:
            self.scale = nn.Parameter(torch.tensor(float(adapter_scalar)))
        else:
            self.register_buffer("scale", torch.tensor(float(adapter_scalar)), persistent=False,)

        self.reset_parameters()

    def reset_parameters(self):
        # Matches AdaptFormer's "lora" initialization.
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return x * self.scale.to(dtype=x.dtype)


class Encoder(nn.Module):
    """
    Frozen DINOv3-B/16 + AdaptFormer.

    Controlled setting shared with Frozen DINOv3 / Full FT / VPT:
        - DINOv3-B/16 backbone
        - intermediate blocks [2, 5, 8, 11]
        - same 1/4, 1/8, 1/16, 1/32 feature construction
        - same downstream decoder

    The only adaptation mechanism here is AdaptFormer, inserted in parallel
    with the FFN branch of every DINOv3 Transformer block.
    """

    def __init__(
        self,
        repo_dir,
        weights,
        bottleneck=64,
        adapter_scalar=0.1,
        adapter_dropout=0.1,
        learnable_scalar=False,
        use_bf16=True,
    ):
        super().__init__()

        self.backbone = torch.hub.load(
            repo_or_dir=repo_dir,
            model="dinov3_vitb16",
            source="local",
            weights=weights,
            pretrained=True,
        )

        # AdaptFormer is a parameter-efficient tuning method: freeze the
        # original pretrained ViT/DINOv3 parameters.
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        self.interaction_indexes = [2, 5, 8, 11]
        self.use_bf16 = use_bf16

        embed_dim = self.backbone.embed_dim
        depth = len(self.backbone.blocks)

        # One parallel AdaptFormer branch for each Transformer block.
        self.adapters = nn.ModuleList([
            AdaptFormerAdapter(
                d_model=embed_dim,
                bottleneck=bottleneck,
                dropout=adapter_dropout,
                adapter_scalar=adapter_scalar,
                learnable_scalar=learnable_scalar,
            )
            for _ in range(depth)
        ])

        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),)
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),)

    def train(self, mode: bool = True):
        # Put trainable AdaptFormer branches into train/eval mode normally,
        # but keep the frozen DINOv3 backbone deterministic in eval mode.
        super().train(mode)
        self.backbone.eval()
        self.adapters.train(mode)
        return self

    @staticmethod
    def _forward_block_with_adapter(block, adapter, x, rope_sincos):
        """
        DINOv3 block with AdaptFormer's parallel FFN branch.

        Original frozen DINOv3 block (eval path):
            x_attn = x + ls1(attn(norm1(x)))
            x_out  = x_attn + ls2(mlp(norm2(x_attn)))

        AdaptFormer parallel form:
            adapt  = Adapter(x_attn)
            x_out  = x_attn + ls2(mlp(norm2(x_attn))) + adapt

        This matches the official AdaptFormer parallel design while keeping
        DINOv3's native attention, RoPE, normalization and FFN intact.
        """
        x_attn = x + block.ls1(
            block.attn(
                block.norm1(x),
                rope=rope_sincos,
            )
        )

        adapt_x = adapter(x_attn)

        ffn_x = block.ls2(
            block.mlp(
                block.norm2(x_attn)
            )
        )

        return x_attn + ffn_x + adapt_x

    def _extract_patch_feature(self, tokens, H, W):
        """
        Apply the same final DINOv3 normalization used by
        get_intermediate_layers(..., norm=True), remove CLS/storage tokens,
        and reshape patch tokens to BCHW.
        """
        prefix_tokens = 1 + self.backbone.n_storage_tokens
        patch_tokens = tokens[:, prefix_tokens:, :]

        # LayerNorm is token-wise, so applying it only to patch tokens is
        # equivalent to normalizing the full sequence and slicing patches.
        patch_tokens = self.backbone.norm(patch_tokens)

        B, N, C = patch_tokens.shape
        expected = H * W
        if N != expected:
            raise RuntimeError(
                f"Unexpected number of patch tokens: got {N}, expected {expected} "
                f"for spatial size {H}x{W}."
            )

        patch_tokens = patch_tokens.reshape(B, H, W, C)
        patch_tokens = patch_tokens.permute(0, 3, 1, 2).contiguous()
        return patch_tokens

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) != 3:
            raise ValueError(
                f"Expected 1-channel or 3-channel input, got {x.size(1)} channels."
            )

        x = (x - self.img_mean) / self.img_std

        if self.use_bf16 and x.is_cuda:
            amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16,)
        else:
            amp_context = contextlib.nullcontext()

        # IMPORTANT: no torch.no_grad() here.
        # DINOv3 weights are frozen, but gradients must pass through the
        # frozen blocks to train the AdaptFormer branches inserted earlier.
        with amp_context:
            tokens, (H, W) = self.backbone.prepare_tokens_with_masks(x)

            if self.backbone.rope_embed is not None:
                rope_sincos = self.backbone.rope_embed(H=H, W=W)
            else:
                rope_sincos = None

            selected = []

            for i, (block, adapter) in enumerate(
                zip(self.backbone.blocks, self.adapters)
            ):
                tokens = self._forward_block_with_adapter(
                    block,
                    adapter,
                    tokens,
                    rope_sincos,
                )

                if i in self.interaction_indexes:
                    selected.append(
                        self._extract_patch_feature(tokens, H, W)
                    )

        if len(selected) != 4:
            raise RuntimeError(
                f"Expected four intermediate features from blocks "
                f"{self.interaction_indexes}, but got {len(selected)}."
            )

        x1, x2, x3, x4 = selected

        x1 = F.interpolate(x1, size=(4 * H, 4 * W), mode="bilinear", align_corners=False,)
        x2 = F.interpolate(x2, size=(2 * H, 2 * W), mode="bilinear", align_corners=False,)
        x3 = F.interpolate(x3, size=(H, W), mode="bilinear", align_corners=False,)
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


class DINOv3AdaptFormerSeg(nn.Module):
    def __init__(
        self,
        num_classes,
        repo_dir,
        weights,
        bottleneck=64,
        adapter_scalar=0.1,
        adapter_dropout=0.1,
        learnable_scalar=False,
        use_bf16=True,
    ):
        super().__init__()

        self.encoder = Encoder(
            repo_dir=repo_dir,
            weights=weights,
            bottleneck=bottleneck,
            adapter_scalar=adapter_scalar,
            adapter_dropout=adapter_dropout,
            learnable_scalar=learnable_scalar,
            use_bf16=use_bf16,
        )

        self.decoder = Decoder(
            num_classes=num_classes,
            encoder_dim=self.encoder.backbone.embed_dim,
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# Keep the same alias as your existing training scripts.
Dino_seg = DINOv3AdaptFormerSeg


if __name__ == "__main__":
    model = DINOv3AdaptFormerSeg(
        num_classes=9,
        bottleneck=64,
        adapter_scalar=0.1,
        adapter_dropout=0.1,
        use_bf16=False,
    )

    x = torch.randn(1, 3, 224, 224)
    y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dino_params = sum(p.numel() for p in model.encoder.backbone.parameters())
    trainable_dino_params = sum(p.numel() for p in model.encoder.backbone.parameters() if p.requires_grad)
    adapter_params = sum(p.numel() for p in model.encoder.adapters.parameters())
    trainable_adapter_params = sum(p.numel() for p in model.encoder.adapters.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in model.decoder.parameters())

    print("-" * 64)
    print(f"Total params              : {total_params:,}")
    print(f"Total trainable params    : {trainable_params:,}")
    print(f"DINOv3 params             : {dino_params:,}")
    print(f"Trainable DINOv3 params   : {trainable_dino_params:,}")
    print(f"AdaptFormer params        : {adapter_params:,}")
    print(f"Trainable adapter params  : {trainable_adapter_params:,}")
    print(f"Decoder params            : {decoder_params:,}")
