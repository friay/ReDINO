from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from utils.ms_deform_attn import MSDeformAttn


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for H_, W_ in spatial_shapes:
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            indexing="ij",
        )
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), dim=-1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, dim=1)
    return reference_points[:, :, None]


def deform_inputs(x, patch_size=16):
    """Deformable-attention geometry used by ViT-CoMer."""
    _, _, h, w = x.shape

    spatial_shapes = torch.as_tensor(
        [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)],
        dtype=torch.long,
        device=x.device,
    )
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )
    reference_points = get_reference_points(
        [(h // patch_size, w // patch_size)], x.device
    )
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor(
        [(h // patch_size, w // patch_size)],
        dtype=torch.long,
        device=x.device,
    )
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )
    reference_points = get_reference_points(
        [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)],
        x.device,
    )
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

    return deform_inputs1, deform_inputs2


def deform_inputs_only_one(device, h, w):
    """Three-level self-interaction geometry for CNN pyramid tokens."""
    spatial_shapes = torch.as_tensor(
        [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)],
        dtype=torch.long,
        device=device,
    )
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )
    reference_points = get_reference_points(
        [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)],
        device=device,
    )
    return [reference_points, spatial_shapes, level_start_index]


class DWConv(nn.Module):
    """Shared 3x3 depthwise convolution used by the CTI ConvFFN."""

    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        if 21 * n != N:
            raise ValueError(f"Expected CNN token ratio 16:4:1, got N={N}.")

        x1 = x[:, : 16 * n].transpose(1, 2).reshape(B, C, H * 2, W * 2)
        x2 = x[:, 16 * n : 20 * n].transpose(1, 2).reshape(B, C, H, W)
        x3 = x[:, 20 * n :].transpose(1, 2).reshape(B, C, H // 2, W // 2)

        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        return torch.cat([x1, x2, x3], dim=1)


class MultiDWConv(nn.Module):
    """Official MRFP multi-receptive-field depthwise convolution."""

    def __init__(self, dim=768):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("MRFP hidden dimension must be even.")

        full_dim = dim
        half_dim = dim // 2

        self.dwconv1 = nn.Conv2d(half_dim, half_dim, 3, 1, 1, groups=half_dim, bias=True)
        self.dwconv2 = nn.Conv2d(half_dim, half_dim, 5, 1, 2, groups=half_dim, bias=True)
        self.dwconv3 = nn.Conv2d(half_dim, half_dim, 3, 1, 1, groups=half_dim, bias=True)
        self.dwconv4 = nn.Conv2d(half_dim, half_dim, 5, 1, 2, groups=half_dim, bias=True)
        self.dwconv5 = nn.Conv2d(half_dim, half_dim, 3, 1, 1, groups=half_dim, bias=True)
        self.dwconv6 = nn.Conv2d(half_dim, half_dim, 5, 1, 2, groups=half_dim, bias=True)

        self.bn1 = nn.BatchNorm2d(full_dim)
        self.bn2 = nn.BatchNorm2d(full_dim)
        self.bn3 = nn.BatchNorm2d(full_dim)
        self.act1 = nn.GELU()
        self.act2 = nn.GELU()
        self.act3 = nn.GELU()

    @staticmethod
    def _split_scale(x, H, W):
        B, N, C = x.shape
        n = N // 21
        if 21 * n != N:
            raise ValueError(f"Expected CNN token ratio 16:4:1, got N={N}.")
        x1 = x[:, : 16 * n].transpose(1, 2).reshape(B, C, H * 2, W * 2)
        x2 = x[:, 16 * n : 20 * n].transpose(1, 2).reshape(B, C, H, W)
        x3 = x[:, 20 * n :].transpose(1, 2).reshape(B, C, H // 2, W // 2)
        return x1, x2, x3

    def forward(self, x, H, W):
        x1, x2, x3 = self._split_scale(x, H, W)
        C = x1.shape[1]

        x11, x12 = x1[:, : C // 2], x1[:, C // 2 :]
        x1 = torch.cat([self.dwconv1(x11), self.dwconv2(x12)], dim=1)
        x1 = self.act1(self.bn1(x1)).flatten(2).transpose(1, 2)

        x21, x22 = x2[:, : C // 2], x2[:, C // 2 :]
        x2 = torch.cat([self.dwconv3(x21), self.dwconv4(x22)], dim=1)
        x2 = self.act2(self.bn2(x2)).flatten(2).transpose(1, 2)

        x31, x32 = x3[:, : C // 2], x3[:, C // 2 :]
        x3 = torch.cat([self.dwconv5(x31), self.dwconv6(x32)], dim=1)
        x3 = self.act3(self.bn3(x3)).flatten(2).transpose(1, 2)

        return torch.cat([x1, x2, x3], dim=1)


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class MRFP(nn.Module):
    """Multi-Receptive Field Feature Pyramid block from ViT-CoMer."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = MultiDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class MultiscaleExtractor(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=12,
        n_points=4,
        n_levels=3,
        deform_ratio=0.5,
        with_cffn=True,
        cffn_ratio=0.25,
        drop=0.0,
        drop_path=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        with_cp=False,
    ):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim,
            n_levels=n_levels,
            n_heads=num_heads,
            n_points=n_points,
            ratio=deform_ratio,
        )
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        def _inner_forward(query, feat):
            attn = self.attn(
                self.query_norm(query),
                reference_points,
                self.feat_norm(feat),
                spatial_shapes,
                level_start_index,
                None,
            )
            query = query + attn
            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query

        if self.with_cp and query.requires_grad:
            return cp.checkpoint(_inner_forward, query, feat, use_reentrant=False)
        return _inner_forward(query, feat)


class CTIToV(nn.Module):
    """CNN -> ViT interaction in the official ViT-CoMer CTI block."""

    def __init__(
        self,
        dim,
        num_heads=12,
        n_points=4,
        n_levels=3,
        deform_ratio=0.5,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.0,
        with_cp=False,
        drop=0.0,
        drop_path=0.0,
        cffn_ratio=0.25,
    ):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim,
            n_levels=n_levels,
            n_heads=num_heads,
            n_points=n_points,
            ratio=deform_ratio,
        )
        self.gamma = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        self.ffn = ConvFFN(dim, hidden_features=int(dim * cffn_ratio), drop=drop)
        self.ffn_norm = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        def _inner_forward(query, feat):
            B, _, C = feat.shape
            c1 = self.attn(
                self.query_norm(feat),
                reference_points,
                self.feat_norm(feat),
                spatial_shapes,
                level_start_index,
                None,
            )
            c1 = c1 + self.drop_path(self.ffn(self.ffn_norm(c1), H, W))

            n_high = H * W * 4
            n_mid = H * W
            c_high = c1[:, :n_high]
            c_mid = c1[:, n_high : n_high + n_mid]
            c_low = c1[:, n_high + n_mid :]

            c_high = F.interpolate(
                c_high.transpose(1, 2).reshape(B, C, H * 2, W * 2),
                scale_factor=0.5,
                mode="bilinear",
                align_corners=False,
            ).flatten(2).transpose(1, 2)
            c_low = F.interpolate(
                c_low.transpose(1, 2).reshape(B, C, H // 2, W // 2),
                scale_factor=2.0,
                mode="bilinear",
                align_corners=False,
            ).flatten(2).transpose(1, 2)

            return query + self.gamma * (c_high + c_mid + c_low)

        if self.with_cp and query.requires_grad:
            return cp.checkpoint(_inner_forward, query, feat, use_reentrant=False)
        return _inner_forward(query, feat)


class CTIToC(nn.Module):
    """ViT -> CNN interaction in ViT-CoMer."""

    def __init__(
        self,
        dim,
        num_heads=12,
        n_points=4,
        deform_ratio=0.5,
        with_cffn=True,
        cffn_ratio=0.25,
        drop=0.0,
        drop_path=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        with_cp=False,
        cnn_feature_interaction=True,
    ):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.with_cp = with_cp
        self.cnn_feature_interaction = cnn_feature_interaction

        if cnn_feature_interaction:
            self.cfinter = MultiscaleExtractor(
                dim=dim,
                n_levels=3,
                num_heads=num_heads,
                n_points=n_points,
                norm_layer=norm_layer,
                deform_ratio=deform_ratio,
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                drop=drop,
                drop_path=drop_path,
                with_cp=with_cp,
            )

    def forward(self, query, feat, H, W):
        def _inner_forward(query, feat):
            _, N, _ = query.shape
            n = N // 21
            if 21 * n != N:
                raise ValueError(f"Expected CNN token ratio 16:4:1, got N={N}.")

            x1 = query[:, : 16 * n]
            x2 = query[:, 16 * n : 20 * n]
            x3 = query[:, 20 * n :]

            if x2.shape[1] != feat.shape[1]:
                raise ValueError(
                    f"Middle CNN scale has {x2.shape[1]} tokens but ViT has {feat.shape[1]}."
                )
            x2 = x2 + feat
            query = torch.cat([x1, x2, x3], dim=1)

            if self.cnn_feature_interaction:
                deform_input = deform_inputs_only_one(query.device, H * 16, W * 16)
                query = self.cfinter(
                    query=self.query_norm(query),
                    reference_points=deform_input[0],
                    feat=self.feat_norm(query),
                    spatial_shapes=deform_input[1],
                    level_start_index=deform_input[2],
                    H=H,
                    W=W,
                )
            return query

        if self.with_cp and query.requires_grad:
            return cp.checkpoint(_inner_forward, query, feat, use_reentrant=False)
        return _inner_forward(query, feat)


class ExtractorCTI(nn.Module):
    """Extra CTI refinement used at the final ViT-CoMer stage."""

    def __init__(
        self,
        dim,
        num_heads=12,
        n_points=4,
        deform_ratio=0.5,
        with_cffn=True,
        cffn_ratio=0.25,
        drop=0.0,
        drop_path=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        with_cp=False,
        cnn_feature_interaction=True,
    ):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        self.cnn_feature_interaction = cnn_feature_interaction

        if with_cffn:
            self.ffn = ConvFFN(dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        if cnn_feature_interaction:
            self.cfinter = MultiscaleExtractor(
                dim=dim,
                n_levels=3,
                num_heads=num_heads,
                n_points=n_points,
                norm_layer=norm_layer,
                deform_ratio=deform_ratio,
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                drop=drop,
                drop_path=drop_path,
                with_cp=with_cp,
            )

    def forward(self, query, feat, H, W):
        def _inner_forward(query, feat):
            _, N, _ = query.shape
            n = N // 21
            if 21 * n != N:
                raise ValueError(f"Expected CNN token ratio 16:4:1, got N={N}.")

            x1 = query[:, : 16 * n]
            x2 = query[:, 16 * n : 20 * n]
            x3 = query[:, 20 * n :]
            x2 = x2 + feat
            query = torch.cat([x1, x2, x3], dim=1)

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))

            if self.cnn_feature_interaction:
                deform_input = deform_inputs_only_one(query.device, H * 16, W * 16)
                query = self.cfinter(
                    query=self.query_norm(query),
                    reference_points=deform_input[0],
                    feat=self.feat_norm(query),
                    spatial_shapes=deform_input[1],
                    level_start_index=deform_input[2],
                    H=H,
                    W=W,
                )
            return query

        if self.with_cp and query.requires_grad:
            return cp.checkpoint(_inner_forward, query, feat, use_reentrant=False)
        return _inner_forward(query, feat)


class CTIBlockDINOv3(nn.Module):
    """One ViT-CoMer interaction stage adapted to DINOv3 blocks."""

    def __init__(
        self,
        dim,
        num_heads=12,
        n_points=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop=0.0,
        drop_path=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        init_values=0.0,
        deform_ratio=0.5,
        extra_cti=False,
        extra_num=4,
        with_cp=False,
        use_cti_to_v=True,
        use_cti_to_c=True,
        dim_ratio=6.0,
        cnn_feature_interaction=True,
    ):
        super().__init__()
        self.use_cti_to_v = use_cti_to_v
        self.use_cti_to_c = use_cti_to_c

        if use_cti_to_v:
            self.cti_to_v = CTIToV(
                dim=dim,
                n_levels=3,
                num_heads=num_heads,
                init_values=init_values,
                n_points=n_points,
                norm_layer=norm_layer,
                deform_ratio=deform_ratio,
                with_cp=with_cp,
                drop=drop,
                drop_path=drop_path,
                cffn_ratio=cffn_ratio,
            )

        if use_cti_to_c:
            self.cti_to_c = CTIToC(
                dim=dim,
                num_heads=num_heads,
                n_points=n_points,
                norm_layer=norm_layer,
                deform_ratio=deform_ratio,
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                drop=drop,
                drop_path=drop_path,
                with_cp=with_cp,
                cnn_feature_interaction=cnn_feature_interaction,
            )

        if extra_cti:
            self.extra_ctis = nn.ModuleList([
                ExtractorCTI(
                    dim=dim,
                    num_heads=num_heads,
                    n_points=n_points,
                    norm_layer=norm_layer,
                    deform_ratio=deform_ratio,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    drop=drop,
                    drop_path=drop_path,
                    with_cp=with_cp,
                    cnn_feature_interaction=cnn_feature_interaction,
                )
                for _ in range(extra_num)
            ])
        else:
            self.extra_ctis = None

        self.mrfp = MRFP(dim, hidden_features=int(dim * dim_ratio))

    def forward(self, x, c, special_tokens, blocks, H, W, rope=None):
        # MRFP + CNN -> ViT interaction
        if self.use_cti_to_v:
            c = self.mrfp(c, H, W)
            n_high = H * W * 4
            n_mid = H * W
            c1 = c[:, :n_high]
            c2 = c[:, n_high : n_high + n_mid]
            c3 = c[:, n_high + n_mid :]
            c = torch.cat([c1, c2 + x, c3], dim=1)

            deform_input = deform_inputs_only_one(c.device, H * 16, W * 16)
            x = self.cti_to_v(
                query=x,
                reference_points=deform_input[0],
                feat=c,
                spatial_shapes=deform_input[1],
                level_start_index=deform_input[2],
                H=H,
                W=W,
            )

        # DINOv3 Transformer blocks. Special tokens are kept in the ViT
        # sequence but never enter CTI's spatial deformable attention.
        full_tokens = torch.cat([special_tokens, x], dim=1)
        for blk in blocks:
            full_tokens = blk(full_tokens, rope)

        n_special = special_tokens.shape[1]
        special_tokens = full_tokens[:, :n_special]
        x = full_tokens[:, n_special:]

        # ViT -> CNN interaction
        if self.use_cti_to_c:
            c = self.cti_to_c(query=c, feat=x, H=H, W=W)

        if self.extra_ctis is not None:
            for cti in self.extra_ctis:
                c = cti(query=c, feat=x, H=H, W=W)

        return x, c, special_tokens


class CNN(nn.Module):
    """CNN spatial branch from the official ViT-CoMer implementation."""

    def __init__(self, inplanes=64, embed_dim=768):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, 3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, 3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, 3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, 2 * inplanes, 3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * inplanes, 4 * inplanes, 3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * inplanes, 4 * inplanes, 3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True),
        )

        self.fc1 = nn.Conv2d(inplanes, embed_dim, 1, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, 1, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, 1, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, 1, bias=True)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)

        B, C, _, _ = c1.shape
        c2 = c2.reshape(B, C, -1).transpose(1, 2)
        c3 = c3.reshape(B, C, -1).transpose(1, 2)
        c4 = c4.reshape(B, C, -1).transpose(1, 2)
        return c1, c2, c3, c4
