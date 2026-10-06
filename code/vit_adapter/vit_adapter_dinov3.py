import math
from functools import partial
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .vit_adapter_modules_dinov3 import (
        InteractionBlockDINOv3,
        MSDeformAttn,
        SpatialPriorModule,
        deform_inputs,
    )
except ImportError:
    from vit_adapter_modules_dinov3 import (
        InteractionBlockDINOv3,
        MSDeformAttn,
        SpatialPriorModule,
        deform_inputs,
    )


class ViTAdapterDINOv3(nn.Module):

    def __init__(
        self,
        backbone,
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
        with_cp=False,
        freeze_backbone=True,
    ):
        super().__init__()

        self.backbone = backbone
        self.freeze_backbone = bool(freeze_backbone)
        self._apply_backbone_freeze_state()

        self.interaction_indexes = [tuple(v) for v in interaction_indexes]
        self.add_vit_feature = add_vit_feature
        self.patch_size = int(self.backbone.patch_size)
        self.embed_dim = int(self.backbone.embed_dim)

        if max(ed for _, ed in self.interaction_indexes) >= len(self.backbone.blocks):
            raise ValueError(
                f"interaction_indexes={self.interaction_indexes} exceed the "
                f"{len(self.backbone.blocks)} DINOv3 blocks."
            )

        self.level_embed = nn.Parameter(torch.zeros(3, self.embed_dim))
        self.spm = SpatialPriorModule(
            inplanes=conv_inplane,
            embed_dim=self.embed_dim,
            with_cp=False,
        )

        self.interactions = nn.ModuleList([
            InteractionBlockDINOv3(
                dim=self.embed_dim,
                num_heads=deform_num_heads,
                n_points=n_points,
                init_values=init_values,
                drop_path=drop_path_rate,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                deform_ratio=deform_ratio,
                extra_extractor=(
                    use_extra_extractor and i == len(self.interaction_indexes) - 1
                ),
                with_cp=with_cp,
            )
            for i in range(len(self.interaction_indexes))
        ])

        self.up = nn.ConvTranspose2d(
            self.embed_dim, self.embed_dim, kernel_size=2, stride=2
        )

        self.norm1 = nn.SyncBatchNorm(self.embed_dim)
        self.norm2 = nn.SyncBatchNorm(self.embed_dim)
        self.norm3 = nn.SyncBatchNorm(self.embed_dim)
        self.norm4 = nn.SyncBatchNorm(self.embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.interactions.apply(self._init_deform_weights)
        torch.nn.init.normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @staticmethod
    def _init_deform_weights(m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        return (
            c2 + self.level_embed[0],
            c3 + self.level_embed[1],
            c4 + self.level_embed[2],
        )

    def _apply_backbone_freeze_state(self):
        """Apply ``freeze_backbone`` to DINOv3 parameters and module mode."""
        self.backbone.requires_grad_(not self.freeze_backbone)
        if self.freeze_backbone:
            self.backbone.eval()

    def set_freeze_backbone(self, freeze: bool = True):
        """
        Switch between the two experimental settings without rebuilding model.

        Args:
            freeze: True freezes all original DINOv3 parameters; False makes
                    them trainable again. ViT-Adapter parameters are unaffected.
        """
        self.freeze_backbone = bool(freeze)
        self._apply_backbone_freeze_state()

        # Respect the current parent module mode after switching.
        if not self.freeze_backbone:
            self.backbone.train(self.training)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        # ``super().train`` changes every submodule. Force only a frozen
        # DINOv3 back to eval mode. In fine-tuning mode it follows ``mode``.
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x):
        B, _, image_h, image_w = x.shape

        if image_h % 32 != 0 or image_w % 32 != 0:
            raise ValueError(
                "ViT-Adapter's 1/8-1/16-1/32 spatial prior requires H and W "
                f"to be divisible by 32. Got {image_h}x{image_w}."
            )

        deform_inputs1, deform_inputs2 = deform_inputs(
            x, patch_size=self.patch_size
        )

        # CNN spatial prior
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # DINOv3 token preparation. This retains CLS + storage tokens and the
        # DINOv3 RoPE geometry instead of pretending DINOv3 is the original
        # absolute-position-embedding ViT used in the 2023 ViT-Adapter code.
        tokens, (H, W) = self.backbone.prepare_tokens_with_masks(x, masks=None)
        num_patch_tokens = H * W
        n_special = tokens.shape[1] - num_patch_tokens
        if n_special < 0:
            raise RuntimeError("Invalid DINOv3 token layout.")

        special_tokens = tokens[:, :n_special]
        patch_tokens = tokens[:, n_special:]

        rope = None
        if getattr(self.backbone, "rope_embed", None) is not None:
            rope = self.backbone.rope_embed(H=H, W=W)

        outs = []

        for i, interaction in enumerate(self.interactions):
            st, ed = self.interaction_indexes[i]
            blocks = self.backbone.blocks[st : ed + 1]

            patch_tokens, c, special_tokens = interaction(
                patch_tokens,
                c,
                special_tokens,
                blocks,
                deform_inputs1,
                deform_inputs2,
                H,
                W,
                rope=rope,
            )

            outs.append(
                patch_tokens.transpose(1, 2)
                .reshape(B, self.embed_dim, H, W)
                .contiguous()
            )

        # Restore the CNN token pyramid.
        n2 = c2.shape[1]
        n3 = c3.shape[1]
        c2_out = c[:, :n2]
        c3_out = c[:, n2 : n2 + n3]
        c4_out = c[:, n2 + n3 :]

        c2_out = (
            c2_out.transpose(1, 2)
            .reshape(B, self.embed_dim, image_h // 8, image_w // 8)
            .contiguous()
        )
        c3_out = (
            c3_out.transpose(1, 2)
            .reshape(B, self.embed_dim, image_h // 16, image_w // 16)
            .contiguous()
        )
        c4_out = (
            c4_out.transpose(1, 2)
            .reshape(B, self.embed_dim, image_h // 32, image_w // 32)
            .contiguous()
        )

        c1_out = self.up(c2_out) + c1

        # As in the original ViT-Adapter, fuse the four ViT stage outputs back
        # into the 1/4, 1/8, 1/16 and 1/32 feature maps.
        if self.add_vit_feature:
            if len(outs) != 4:
                raise RuntimeError(
                    "add_vit_feature=True expects exactly four interaction stages."
                )
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(
                x1, size=c1_out.shape[-2:], mode="bilinear", align_corners=False
            )
            x2 = F.interpolate(
                x2, size=c2_out.shape[-2:], mode="bilinear", align_corners=False
            )
            x3 = F.interpolate(
                x3, size=c3_out.shape[-2:], mode="bilinear", align_corners=False
            )
            x4 = F.interpolate(
                x4, size=c4_out.shape[-2:], mode="bilinear", align_corners=False
            )

            c1_out = c1_out + x1
            c2_out = c2_out + x2
            c3_out = c3_out + x3
            c4_out = c4_out + x4

        f1 = self.norm1(c1_out)
        f2 = self.norm2(c2_out)
        f3 = self.norm3(c3_out)
        f4 = self.norm4(c4_out)

        return f1, f2, f3, f4
