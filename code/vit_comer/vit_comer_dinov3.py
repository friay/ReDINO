"""ViT-CoMer backbone adapted to a DINOv3 ViT-B/16 backbone."""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .comer_modules_dinov3 import CNN, CTIBlockDINOv3, MSDeformAttn
except ImportError:
    from comer_modules_dinov3 import CNN, CTIBlockDINOv3, MSDeformAttn


class ViTCoMerDINOv3(nn.Module):
    """
    ViT-CoMer instantiated on DINOv3-B/16.

    The official ViT-CoMer architecture is preserved:
      * CNN spatial branch
      * MRFP before CNN -> ViT fusion
      * bidirectional CTI interaction
      * four interaction stages: [0,2], [3,5], [6,8], [9,11]
      * four extra CTI refinements at the final stage (segmentation version)
      * stage-wise ViT feature addition to the four output pyramid levels

    DINOv3-specific adaptation:
      * CLS/storage tokens stay inside DINOv3 Transformer blocks
      * CTI only processes spatial patch tokens
      * DINOv3 RoPE is preserved

    freeze_backbone=False is the default because the original ViT-CoMer
    fine-tunes its pretrained ViT backbone together with the new modules.
    """

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
        use_extra_cti=True,
        extra_num=4,
        use_cti_to_v=(True, True, True, True),
        use_cti_to_c=(True, True, True, True),
        cnn_feature_interaction=(True, True, True, True),
        dim_ratio=6.0,
        with_cp=False,
        freeze_backbone=False,
    ):
        super().__init__()

        self.backbone = backbone
        self.freeze_backbone = bool(freeze_backbone)
        self.interaction_indexes = [tuple(v) for v in interaction_indexes]
        self.add_vit_feature = bool(add_vit_feature)
        self.patch_size = int(self.backbone.patch_size)
        self.embed_dim = int(self.backbone.embed_dim)

        self._apply_backbone_freeze_state()

        if self.patch_size != 16:
            raise ValueError(
                "This ViT-CoMer adaptation follows the official stride-16 "
                f"geometry, but got patch_size={self.patch_size}."
            )
        if max(ed for _, ed in self.interaction_indexes) >= len(self.backbone.blocks):
            raise ValueError(
                f"interaction_indexes={self.interaction_indexes} exceed "
                f"the {len(self.backbone.blocks)} DINOv3 blocks."
            )

        def stage_value(v, i):
            return v if isinstance(v, bool) else bool(v[i])

        self.level_embed = nn.Parameter(torch.zeros(3, self.embed_dim))
        self.spm = CNN(inplanes=conv_inplane, embed_dim=self.embed_dim)

        self.interactions = nn.ModuleList([
            CTIBlockDINOv3(
                dim=self.embed_dim,
                num_heads=deform_num_heads,
                n_points=n_points,
                init_values=init_values,
                drop_path=drop_path_rate,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                deform_ratio=deform_ratio,
                use_cti_to_v=stage_value(use_cti_to_v, i),
                use_cti_to_c=stage_value(use_cti_to_c, i),
                cnn_feature_interaction=stage_value(cnn_feature_interaction, i),
                dim_ratio=dim_ratio,
                extra_cti=(use_extra_cti and i == len(self.interaction_indexes) - 1),
                extra_num=extra_num,
                with_cp=with_cp,
            )
            for i in range(len(self.interaction_indexes))
        ])

        self.up = nn.ConvTranspose2d(self.embed_dim, self.embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(self.embed_dim)
        self.norm2 = nn.SyncBatchNorm(self.embed_dim)
        self.norm3 = nn.SyncBatchNorm(self.embed_dim)
        self.norm4 = nn.SyncBatchNorm(self.embed_dim)

        # Initialize only newly introduced ViT-CoMer modules.
        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.interactions.apply(self._init_deform_weights)
        torch.nn.init.normal_(self.level_embed)

    def _apply_backbone_freeze_state(self):
        self.backbone.requires_grad_(not self.freeze_backbone)
        if self.freeze_backbone:
            self.backbone.eval()

    def set_freeze_backbone(self, freeze: bool = True):
        self.freeze_backbone = bool(freeze)
        self._apply_backbone_freeze_state()
        if not self.freeze_backbone:
            self.backbone.train(self.training)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
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

    def _prepare_dinov3_tokens(self, x):
        prepared = self.backbone.prepare_tokens_with_masks(x, masks=None)

        if isinstance(prepared, tuple):
            tokens = prepared[0]
            hw = prepared[1] if len(prepared) > 1 else None
        else:
            tokens = prepared
            hw = None

        if isinstance(hw, (tuple, list)) and len(hw) == 2:
            H, W = int(hw[0]), int(hw[1])
        else:
            H = x.shape[-2] // self.patch_size
            W = x.shape[-1] // self.patch_size

        num_patch_tokens = H * W
        n_special = tokens.shape[1] - num_patch_tokens
        if n_special < 0:
            raise RuntimeError(
                f"Invalid DINOv3 token layout: total={tokens.shape[1]}, "
                f"patch={num_patch_tokens}."
            )

        special_tokens = tokens[:, :n_special]
        patch_tokens = tokens[:, n_special:]

        rope = None
        if getattr(self.backbone, "rope_embed", None) is not None:
            try:
                rope = self.backbone.rope_embed(H=H, W=W)
            except TypeError:
                rope = self.backbone.rope_embed(H, W)

        return patch_tokens, special_tokens, H, W, rope

    def forward(self, x):
        B, _, image_h, image_w = x.shape

        if image_h % 32 != 0 or image_w % 32 != 0:
            raise ValueError(
                "ViT-CoMer requires H and W divisible by 32 for its "
                f"1/8-1/16-1/32 CNN pyramid. Got {image_h}x{image_w}."
            )

        # CNN branch
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # DINOv3 patch tokens + special tokens + RoPE
        x_tokens, special_tokens, H, W, rope = self._prepare_dinov3_tokens(x)

        if H != image_h // 16 or W != image_w // 16:
            raise RuntimeError(
                f"Unexpected DINOv3 patch grid {H}x{W} for image "
                f"{image_h}x{image_w}."
            )

        stage_vit_features = []

        for i, interaction in enumerate(self.interactions):
            st, ed = self.interaction_indexes[i]
            blocks = self.backbone.blocks[st : ed + 1]

            x_tokens, c, special_tokens = interaction(
                x=x_tokens,
                c=c,
                special_tokens=special_tokens,
                blocks=blocks,
                H=H,
                W=W,
                rope=rope,
            )

            stage_vit_features.append(
                x_tokens.transpose(1, 2)
                .reshape(B, self.embed_dim, H, W)
                .contiguous()
            )

        # Restore CNN token pyramid
        n2 = c2.shape[1]
        n3 = c3.shape[1]
        c2_out = c[:, :n2]
        c3_out = c[:, n2 : n2 + n3]
        c4_out = c[:, n2 + n3 :]

        c2_out = c2_out.transpose(1, 2).reshape(
            B, self.embed_dim, image_h // 8, image_w // 8
        ).contiguous()
        c3_out = c3_out.transpose(1, 2).reshape(
            B, self.embed_dim, image_h // 16, image_w // 16
        ).contiguous()
        c4_out = c4_out.transpose(1, 2).reshape(
            B, self.embed_dim, image_h // 32, image_w // 32
        ).contiguous()

        c1_out = self.up(c2_out) + c1

        if self.add_vit_feature:
            if len(stage_vit_features) != 4:
                raise RuntimeError(
                    "add_vit_feature=True expects exactly four CTI stages."
                )
            x1, x2, x3, x4 = stage_vit_features
            x1 = F.interpolate(x1, size=c1_out.shape[-2:], mode="bilinear", align_corners=False)
            x2 = F.interpolate(x2, size=c2_out.shape[-2:], mode="bilinear", align_corners=False)
            x3 = F.interpolate(x3, size=c3_out.shape[-2:], mode="bilinear", align_corners=False)
            x4 = F.interpolate(x4, size=c4_out.shape[-2:], mode="bilinear", align_corners=False)
            c1_out = c1_out + x1
            c2_out = c2_out + x2
            c3_out = c3_out + x3
            c4_out = c4_out + x4

        f1 = self.norm1(c1_out)
        f2 = self.norm2(c2_out)
        f3 = self.norm3(c3_out)
        f4 = self.norm4(c4_out)
        return f1, f2, f3, f4
