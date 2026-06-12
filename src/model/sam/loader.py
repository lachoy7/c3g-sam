"""Load SAM checkpoints (full model or frozen image encoder only)."""

from __future__ import annotations

import os

import torch.nn as nn
from segment_anything import sam_model_registry
from segment_anything.modeling import Sam

from .constants import SAM_FEATURE_DIM, SAM_MODELS


def load_sam(
    model_variant: str = "sam_vit_h",
    checkpoint_path: str = "./pretrained_weights/sam_vit_h.pth",
    *,
    freeze: bool = True,
) -> Sam:
    """Load the full SAM model from a checkpoint."""
    if model_variant not in SAM_MODELS:
        raise ValueError(
            f"Unsupported SAM model variant '{model_variant}'. "
            f"Supported: {list(SAM_MODELS.keys())}"
        )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"SAM checkpoint not found at '{checkpoint_path}'. "
            f"Please download the SAM weights to this path."
        )

    sam_type = SAM_MODELS[model_variant]
    sam = sam_model_registry[sam_type](checkpoint=checkpoint_path)

    if freeze:
        for param in sam.parameters():
            param.requires_grad = False
        sam.eval()

    return sam


def load_sam_encoder(
    model_variant: str,
    checkpoint_path: str,
) -> tuple[nn.Module, int]:
    """Load a frozen SAM image encoder (used as a foundation feature extractor)."""
    sam = load_sam(model_variant, checkpoint_path, freeze=True)
    return sam.image_encoder, SAM_FEATURE_DIM
