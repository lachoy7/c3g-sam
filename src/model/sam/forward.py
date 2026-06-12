"""Vanilla SAM forward pass (image encoder -> prompt encoder -> mask decoder)."""

from __future__ import annotations

from typing import Optional

import torch
from segment_anything.modeling import Sam

from .preprocess import (
    generate_grid_points,
    input_sizes_from_resized,
    original_sizes_from_images,
    postprocess_masks,
    preprocess_images,
    resize_images_longest_side,
    transform_boxes,
    transform_point_coords,
)


@torch.no_grad()
def encode_images(
    sam: Sam,
    images: torch.Tensor,
    *,
    original_sizes: Optional[list[tuple[int, int]]] = None,
) -> tuple[torch.Tensor, list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Run the SAM image encoder on a batch of RGB images.

    Args:
        sam: Loaded SAM model.
        images: Bx3xHxW float tensor in [0, 255] (RGB).
        original_sizes: Per-image (H, W) before resize; inferred from images if None.

    Returns:
        image_embeddings: Bx256x64x64
        input_sizes: (H, W) after longest-side resize
        original_sizes: (H, W) in the input coordinate frame
    """
    if original_sizes is None:
        original_sizes = original_sizes_from_images(images)

    resized = resize_images_longest_side(images, sam.image_encoder.img_size)
    input_sizes = input_sizes_from_resized(resized)
    preprocessed = preprocess_images(sam, resized)
    image_embeddings = sam.image_encoder(preprocessed)
    return image_embeddings, input_sizes, original_sizes


@torch.no_grad()
def predict_masks(
    sam: Sam,
    image_embeddings: torch.Tensor,
    input_sizes: list[tuple[int, int]],
    original_sizes: list[tuple[int, int]],
    *,
    point_coords: Optional[torch.Tensor] = None,
    point_labels: Optional[torch.Tensor] = None,
    boxes: Optional[torch.Tensor] = None,
    mask_inputs: Optional[torch.Tensor] = None,
    multimask_output: bool = True,
    return_logits: bool = False,
    segment_everything: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Decode masks from precomputed image embeddings and prompts.

    Prompt coordinates are in original-image pixel space (X, Y), matching SamPredictor.
    When no prompts are given and segment_everything=True, uses a fixed point grid.
    """
    batch_size = image_embeddings.shape[0]
    has_prompts = (
        point_coords is not None
        or boxes is not None
        or mask_inputs is not None
        or segment_everything
    )
    if not has_prompts:
        raise ValueError(
            "Provide point_coords, boxes, mask_inputs, or segment_everything=True."
        )

    all_masks: list[torch.Tensor] = []
    all_iou: list[torch.Tensor] = []
    all_low_res: list[torch.Tensor] = []
    image_pe = sam.prompt_encoder.get_dense_pe()

    for i in range(batch_size):
        orig_size = original_sizes[i]
        input_size = input_sizes[i]

        if segment_everything and point_coords is None and boxes is None:
            grid_pts, grid_labels = generate_grid_points(1, image_embeddings.device)
            points = (grid_pts, grid_labels)
            box = None
        else:
            points = None
            if point_coords is not None:
                pc = transform_point_coords(
                    sam, point_coords[i : i + 1], orig_size
                )
                pl = point_labels[i : i + 1] if point_labels is not None else None
                points = (pc, pl)
            box = (
                transform_boxes(sam, boxes[i : i + 1], orig_size)
                if boxes is not None
                else None
            )

        mask_in = mask_inputs[i : i + 1] if mask_inputs is not None else None
        sparse_emb, dense_emb = sam.prompt_encoder(
            points=points, boxes=box, masks=mask_in
        )
        low_res_masks, iou_predictions = sam.mask_decoder(
            image_embeddings=image_embeddings[i : i + 1],
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=multimask_output,
        )
        masks = postprocess_masks(sam, low_res_masks, input_size, orig_size)
        if not return_logits:
            masks = masks > sam.mask_threshold

        all_masks.append(masks)
        all_iou.append(iou_predictions)
        all_low_res.append(low_res_masks)

    return {
        "masks": torch.cat(all_masks, dim=0),
        "iou_predictions": torch.cat(all_iou, dim=0),
        "low_res_logits": torch.cat(all_low_res, dim=0),
    }


@torch.no_grad()
def forward(
    sam: Sam,
    images: torch.Tensor,
    *,
    point_coords: Optional[torch.Tensor] = None,
    point_labels: Optional[torch.Tensor] = None,
    boxes: Optional[torch.Tensor] = None,
    mask_inputs: Optional[torch.Tensor] = None,
    multimask_output: bool = True,
    return_logits: bool = False,
    segment_everything: bool = False,
    original_sizes: Optional[list[tuple[int, int]]] = None,
) -> dict[str, torch.Tensor]:
    """
    End-to-end vanilla SAM: preprocess, encode, prompt, decode, postprocess.

    This mirrors segment_anything.modeling.Sam.forward / SamPredictor.predict_torch.
    """
    image_embeddings, input_sizes, orig_sizes = encode_images(
        sam, images, original_sizes=original_sizes
    )
    return predict_masks(
        sam,
        image_embeddings,
        input_sizes,
        orig_sizes,
        point_coords=point_coords,
        point_labels=point_labels,
        boxes=boxes,
        mask_inputs=mask_inputs,
        multimask_output=multimask_output,
        return_logits=return_logits,
        segment_everything=segment_everything,
    )
