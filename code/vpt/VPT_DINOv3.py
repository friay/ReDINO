import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepVPTDINOv3Encoder(nn.Module):
    """
    DINOv3-B/16 + Visual Prompt Tuning (VPT).

    Default configuration uses VPT-Deep:
      - DINOv3 backbone parameters are frozen.
      - A distinct set of prompt tokens is inserted before each transformer block.
      - Prompt tokens are treated as prefix tokens, while only patch tokens receive 2D RoPE.
      - Intermediate patch features are collected from blocks [2, 5, 8, 11].

    """

    def __init__(self, repo_dir, weights, num_prompt_tokens=10, deep=True, use_bf16=True,):
        super().__init__()

        self.backbone = torch.hub.load(
            repo_or_dir=repo_dir,
            model="dinov3_vitb16",
            source="local",
            weights=weights,
            pretrained=True,
        )

        # VPT freezes the pretrained backbone.
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        self.interaction_indexes = [2, 5, 8, 11]
        self.num_prompt_tokens = int(num_prompt_tokens)
        self.deep = bool(deep)
        self.use_bf16 = bool(use_bf16)

        self.embed_dim = int(self.backbone.embed_dim)
        self.depth = len(self.backbone.blocks)

        if self.num_prompt_tokens <= 0:
            raise ValueError("num_prompt_tokens must be > 0")

        # VPT-Shallow: one prompt set shared through all blocks.
        # VPT-Deep: an independent prompt set for every transformer block.
        if self.deep:
            self.prompt_embeddings = nn.Parameter(torch.empty(self.depth, self.num_prompt_tokens, self.embed_dim))
        else:
            self.prompt_embeddings = nn.Parameter(torch.empty(1, self.num_prompt_tokens, self.embed_dim))

        self._init_prompt_embeddings()

        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),)
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),)

    def _init_prompt_embeddings(self):
        # Small Xavier-style initialization suitable for prompt tokens.
        val = math.sqrt(6.0 / float(3 * 16 * 16 + self.embed_dim))
        nn.init.uniform_(self.prompt_embeddings, -val, val)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the frozen DINOv3 backbone in eval mode even when the whole model
        # is switched to train(). Prompt parameters and decoder remain trainable.
        self.backbone.eval()
        return self

    def _insert_prompt(self, x, prompt, prefix_len):
        """
        x layout before insertion:
            [CLS, storage tokens, patch tokens]

        x layout after insertion:
            [CLS, storage tokens, prompt tokens, patch tokens]
        """
        B = x.shape[0]
        prompt = prompt.unsqueeze(0).expand(B, -1, -1)
        prompt = prompt.to(dtype=x.dtype, device=x.device)

        return torch.cat(
            [
                x[:, :prefix_len, :],
                prompt,
                x[:, prefix_len:, :],
            ],
            dim=1,
        )

    def _replace_prompt(self, x, prompt, prefix_len):
        """
        Replace the previous block's prompt tokens with a new prompt set.
        This follows the VPT-Deep idea: prompts are replaced rather than
        accumulated across transformer blocks.
        """
        B = x.shape[0]
        prompt = prompt.unsqueeze(0).expand(B, -1, -1)
        prompt = prompt.to(dtype=x.dtype, device=x.device)

        old_prompt_end = prefix_len + self.num_prompt_tokens

        return torch.cat(
            [
                x[:, :prefix_len, :],
                prompt,
                x[:, old_prompt_end:, :],
            ],
            dim=1,
        )

    def _normalize_patch_tokens(self, patch_tokens):
        # get_intermediate_layers(..., norm=True) applies backbone.norm to patch
        # tokens. We reproduce the same behavior here.
        return self.backbone.norm(patch_tokens)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        x = (x - self.img_mean) / self.img_std

        if self.use_bf16 and x.is_cuda:
            amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16,)
        else:
            amp_context = contextlib.nullcontext()

        # IMPORTANT:
        # Do NOT use torch.no_grad() here. Although DINOv3 weights are frozen,
        # gradients must pass through its blocks to optimize the prompt tokens.
        with amp_context:
            tokens, (H, W) = self.backbone.prepare_tokens_with_masks(x)

            # DINOv3 prefix consists of CLS + storage tokens.
            prefix_len = int(self.backbone.n_storage_tokens) + 1

            # Insert the first prompt set.
            tokens = self._insert_prompt(tokens, self.prompt_embeddings[0], prefix_len,)

            outputs = []
            spatial_token_count = H * W

            if self.backbone.rope_embed is not None:
                rope_sincos = self.backbone.rope_embed(H=H, W=W)
            else:
                rope_sincos = None

            for i, blk in enumerate(self.backbone.blocks):
                # For VPT-Deep, replace the prompt tokens before each block after
                # the first one. For shallow VPT, the original prompts propagate.
                if self.deep and i > 0:
                    tokens = self._replace_prompt(
                        tokens,
                        self.prompt_embeddings[i],
                        prefix_len,
                    )

                tokens = blk(tokens, rope_sincos)

                if i in self.interaction_indexes:
                    # The last H*W tokens are always the true spatial patch tokens.
                    patch_tokens = tokens[:, -spatial_token_count:, :]
                    patch_tokens = self._normalize_patch_tokens(patch_tokens)

                    feat = patch_tokens.reshape(
                        patch_tokens.shape[0], H, W, self.embed_dim
                    ).permute(0, 3, 1, 2).contiguous()

                    outputs.append(feat)

        if len(outputs) != 4:
            raise RuntimeError(
                f"Expected 4 intermediate feature maps from blocks "
                f"{self.interaction_indexes}, but got {len(outputs)}."
            )

        x1, x2, x3, x4 = outputs

        x1 = F.interpolate(x1, size=(4 * H, 4 * W), mode="bilinear", align_corners=False,)
        x2 = F.interpolate(x2, size=(2 * H, 2 * W), mode="bilinear", align_corners=False,)
        # x3 stays at H x W.
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

        return self.conv(x)


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


class DINOv3VPTSeg(nn.Module):
    def __init__(
        self,
        num_classes,
        repo_dir,
        weights,
        num_prompt_tokens=10,
        deep=True,
        use_bf16=True,
    ):
        super().__init__()

        self.encoder = DeepVPTDINOv3Encoder(
            repo_dir=repo_dir,
            weights=weights,
            num_prompt_tokens=num_prompt_tokens,
            deep=deep,
            use_bf16=use_bf16,
        )

        self.decoder = Decoder(
            num_classes=num_classes,
            encoder_dim=self.encoder.embed_dim,
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))



if __name__ == "__main__":
    model = DINOv3VPTSeg(
        num_classes=9,
        num_prompt_tokens=10,
        deep=True,
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
    prompt_params = model.encoder.prompt_embeddings.numel()
    decoder_params = sum(p.numel() for p in model.decoder.parameters())

    print("-" * 60)
    print(f"Total params            : {total_params:,}")
    print(f"Trainable params        : {trainable_params:,}")
    print(f"DINOv3 params           : {dino_params:,}")
    print(f"Trainable DINOv3 params : {trainable_dino_params:,}")
    print(f"Prompt params           : {prompt_params:,}")
    print(f"Decoder params          : {decoder_params:,}")
