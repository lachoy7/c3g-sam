"""C3G SAM mask decoder wrapper (precomputed 64x64 embeddings, not full vanilla SAM)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.sam.constants import GRID_SIZE, SAM_MODELS
from src.model.sam.loader import load_sam
from src.model.sam.preprocess import generate_grid_points


class SAMMaskDecoderWrapper(nn.Module):
    """Wraps SAM's mask decoder to accept rendered Gaussian features (Bx256x64x64)."""

    def __init__(
        self, sam_checkpoint, model_variant="sam_vit_h", use_lora=False, lora_rank=4
    ):
        super().__init__()

        if model_variant not in SAM_MODELS:
            raise ValueError(
                f"Unsupported SAM model variant '{model_variant}'. "
                f"Supported: {list(SAM_MODELS.keys())}"
            )

        sam = load_sam(model_variant, sam_checkpoint, freeze=True)
        self.mask_decoder = sam.mask_decoder
        self.prompt_encoder = sam.prompt_encoder

        self.use_lora = use_lora
        if use_lora:
            self.inject_lora(lora_rank)

    def inject_lora(self, rank):
        """Inject LoRA on v_proj layers in token-to-image cross-attention."""
        from src.model.lora import inject_lora

        for layer in self.mask_decoder.transformer.layers:
            inject_lora(layer, "cross_attn_token_to_image.v_proj", rank=rank)

        inject_lora(
            self.mask_decoder.transformer, "final_attn_token_to_image.v_proj", rank=rank
        )

    @staticmethod
    def _normalize_point_prompts(
        point_coords: torch.Tensor | None,
        point_labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if point_coords is None:
            return None
        if point_coords.dim() == 2:
            point_coords = point_coords.unsqueeze(1)
        if point_labels is None:
            raise ValueError("point_labels are required when point_coords are set.")
        if point_labels.dim() == 1:
            point_labels = point_labels.unsqueeze(1)
        return point_coords, point_labels

    def _encode_prompts(
        self,
        batch_size: int,
        device: torch.device,
        *,
        point_coords: torch.Tensor | None = None,
        point_labels: torch.Tensor | None = None,
        box: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        has_points = point_coords is not None or point_labels is not None
        has_box = box is not None
        if has_points and point_coords is None:
            raise ValueError("point_coords are required when point_labels are set.")

        if has_points or has_box:
            points = None
            if has_points:
                point_coords, point_labels = self._normalize_point_prompts(
                    point_coords, point_labels
                )
                points = (point_coords, point_labels)
            return self.prompt_encoder(points=points, boxes=box, masks=None)

        grid_pts, grid_labels = generate_grid_points(
            batch_size, device, grid_size=GRID_SIZE
        )
        return self.prompt_encoder(
            points=(grid_pts, grid_labels), boxes=None, masks=None
        )

    def _forward_single(
        self,
        image_embeddings: torch.Tensor,
        *,
        point_coords: torch.Tensor | None = None,
        point_labels: torch.Tensor | None = None,
        box: torch.Tensor | None = None,
        return_iou_predictions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run mask decoder for one embedding (1, C, 64, 64).

        SAM's ``predict_masks`` repeats image embeddings by ``tokens.shape[0]``,
        which only matches a single-image batch. Callers with B>1 must loop here.
        """
        sparse_emb, dense_emb = self._encode_prompts(
            1,
            image_embeddings.device,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
        )
        image_pe = self.prompt_encoder.get_dense_pe()
        masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=True,
        )
        if return_iou_predictions:
            return masks, iou_predictions
        return masks

    def forward(
        self,
        rendered_features: torch.Tensor,
        point_coords: torch.Tensor | None = None,
        point_labels: torch.Tensor | None = None,
        box: torch.Tensor | None = None,
        return_iou_predictions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the SAM mask decoder on ``rendered_features`` (B, C, H, W).

        Without point/box prompts, uses segment-everything grid prompts per image.
        Batched mask-decoder calls are implemented as per-sample loops because
        SAM's mask decoder repeats embeddings by the prompt batch dimension.

        Returns:
            Low-res mask logits (B, num_multimasks, mask_h, mask_w), and optionally
            SAM IoU head scores (B, num_multimasks) when ``return_iou_predictions``.
        """
        batch_size, _, height, width = rendered_features.shape
        if batch_size == 0:
            raise ValueError("rendered_features must have batch size > 0.")

        if height != 64 or width != 64:
            image_embeddings = F.interpolate(
                rendered_features,
                size=(64, 64),
                mode="bilinear",
                align_corners=False,
            )
        else:
            image_embeddings = rendered_features

        if batch_size == 1:
            return self._forward_single(
                image_embeddings,
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                return_iou_predictions=return_iou_predictions,
            )

        masks = []
        ious = []
        for index in range(batch_size):
            sample_coords = (
                point_coords[index : index + 1] if point_coords is not None else None
            )
            sample_labels = (
                point_labels[index : index + 1] if point_labels is not None else None
            )
            sample_box = box[index : index + 1] if box is not None else None
            if return_iou_predictions:
                sample_masks, sample_ious = self._forward_single(
                    image_embeddings[index : index + 1],
                    point_coords=sample_coords,
                    point_labels=sample_labels,
                    box=sample_box,
                    return_iou_predictions=True,
                )
                masks.append(sample_masks)
                ious.append(sample_ious)
            else:
                masks.append(
                    self._forward_single(
                        image_embeddings[index : index + 1],
                        point_coords=sample_coords,
                        point_labels=sample_labels,
                        box=sample_box,
                    )
                )
        stacked_masks = torch.cat(masks, dim=0)
        if return_iou_predictions:
            return stacked_masks, torch.cat(ious, dim=0)
        return stacked_masks
