"""SAM shared constants (model registry, geometry, normalization)."""

from __future__ import annotations

# Checkpoint key -> segment_anything registry name
SAM_MODELS: dict[str, str] = {
    "sam_vit_h": "vit_h",
    "sam_vit_l": "vit_l",
    "sam_vit_b": "vit_b",
}

SAM_IMAGE_SIZE = 1024
SAM_EMBED_SIZE = 64
SAM_FEATURE_DIM = 256
SAM_MASK_INPUT_SIZE = 256

# ImageNet-style normalization used by the official SAM checkpoint
PIXEL_MEAN = (123.675, 116.28, 103.53)
PIXEL_STD = (58.395, 57.12, 57.375)

# Segment-everything grid density (see SAMMaskDecoderWrapper)
GRID_SIZE = 8
