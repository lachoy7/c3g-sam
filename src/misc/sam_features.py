"""Helpers for precomputed SAM encoder features (256×64×64)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

SAM_FEATURE_CHANNELS = 256
SAM_FEATURE_HW = 64


def downsample_sam_for_encoder(
    sam_features: Tensor,
    h: int,
    w: int,
    patch_size: int,
) -> Tensor:
    """Match InstillTransformer patch-token count (same as distillation training)."""
    b, v, c, fh, fw = sam_features.shape
    patch_h = h // patch_size
    patch_w = w // patch_size
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError(
            f"Cannot downsample SAM features: image ({h}, {w}) is smaller than "
            f"patch_size={patch_size}"
        )
    if fh == patch_h and fw == patch_w:
        return sam_features
    flat = rearrange(sam_features, "b v c h w -> (b v) c h w")
    flat = F.interpolate(
        flat, size=(patch_h, patch_w), mode="bilinear", align_corners=False
    )
    return rearrange(flat, "(b v) c h w -> b v c h w", b=b, v=v)


def reorder_context_target(features: Tensor, num_context_views: int) -> Tensor:
    """Context+target batch order -> target-first (feature_rendering_loss layout)."""
    return torch.cat(
        (features[:, num_context_views:], features[:, :num_context_views]), dim=1
    )
