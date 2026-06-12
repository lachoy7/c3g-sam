"""Image and prompt preprocessing for vanilla SAM (resize, normalize, pad)."""

from __future__ import annotations

from typing import Optional

import torch
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide

from .constants import SAM_IMAGE_SIZE


def get_resize_transform(sam: Sam) -> ResizeLongestSide:
    """Longest-side resize transform matching the loaded SAM checkpoint."""
    return ResizeLongestSide(sam.image_encoder.img_size)


def resize_images_longest_side(
    images: torch.Tensor,
    target_length: int = SAM_IMAGE_SIZE,
) -> torch.Tensor:
    """Resize Bx3xHxW images so the longest side equals target_length."""
    transform = ResizeLongestSide(target_length)
    return transform.apply_image_torch(images.float())


def preprocess_images(sam: Sam, resized_images: torch.Tensor) -> torch.Tensor:
    """Normalize and pad resized images to the SAM encoder input (Bx3x1024x1024)."""
    return torch.stack(
        [sam.preprocess(resized_images[i]) for i in range(resized_images.shape[0])],
        dim=0,
    )


def transform_point_coords(
    sam: Sam,
    point_coords: torch.Tensor,
    original_size: tuple[int, int],
) -> torch.Tensor:
    """Map point prompts from original pixel coords to the resized input frame."""
    transform = get_resize_transform(sam)
    return transform.apply_coords_torch(point_coords, original_size)


def transform_boxes(
    sam: Sam,
    boxes: torch.Tensor,
    original_size: tuple[int, int],
) -> torch.Tensor:
    """Map box prompts (XYXY) from original pixels to the resized input frame."""
    transform = get_resize_transform(sam)
    return transform.apply_boxes_torch(boxes, original_size)


def generate_grid_points(
    batch_size: int,
    device: torch.device,
    *,
    image_size: int = SAM_IMAGE_SIZE,
    grid_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evenly spaced foreground points for segment-everything mode."""
    step = image_size // grid_size
    offset = step // 2
    coords = torch.arange(offset, image_size, step, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    points = points.unsqueeze(0).expand(batch_size, -1, -1)
    labels = torch.ones(batch_size, points.shape[1], device=device, dtype=torch.int)
    return points, labels


def postprocess_masks(
    sam: Sam,
    low_res_masks: torch.Tensor,
    input_size: tuple[int, int],
    original_size: tuple[int, int],
) -> torch.Tensor:
    """Upsample low-res mask logits to the original image resolution."""
    return sam.postprocess_masks(low_res_masks, input_size, original_size)


def original_sizes_from_images(
    images: torch.Tensor,
) -> list[tuple[int, int]]:
    """Infer (H, W) original sizes from a Bx3xHxW tensor."""
    _, _, h, w = images.shape
    return [(h, w)] * images.shape[0]


def input_sizes_from_resized(
    resized_images: torch.Tensor,
) -> list[tuple[int, int]]:
    """Record (H, W) after longest-side resize (before square padding)."""
    _, _, h, w = resized_images.shape
    return [(h, w)] * resized_images.shape[0]
