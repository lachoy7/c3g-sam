"""Prompted BCE + Dice loss with best-of-3 mask selection."""

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.prompt_sampler import PromptSampler, decompose_label_map
from ..model.sam_decoder import SAMMaskDecoderWrapper
from ..model.types import Gaussians
from .loss import Loss

logger = logging.getLogger(__name__)


@dataclass
class LossSegmentationPromptedCfg:
    weight: float
    sam_checkpoint: str = ""
    sam_model_variant: str = "sam_vit_h"
    use_lora: bool = False
    lora_rank: int = 4
    prompt_strategy: str = "centroid"
    min_object_pixels: int = 16


@dataclass
class LossSegmentationPromptedCfgWrapper:
    segmentation_prompted: LossSegmentationPromptedCfg


class LossSegmentationPrompted(
    Loss[LossSegmentationPromptedCfg, LossSegmentationPromptedCfgWrapper]
):
    """Prompted BCE + Dice loss with best-of-3 mask selection."""

    def __init__(self, cfg: LossSegmentationPromptedCfgWrapper):
        super().__init__(cfg)

        self.mask_decoder = SAMMaskDecoderWrapper(
            sam_checkpoint=self.cfg.sam_checkpoint,
            model_variant=self.cfg.sam_model_variant,
            use_lora=self.cfg.use_lora,
            lora_rank=self.cfg.lora_rank,
        )

        self.prompt_sampler = PromptSampler(
            strategy=self.cfg.prompt_strategy,
            min_object_pixels=self.cfg.min_object_pixels,
        )

    def dice_loss(self, pred, target):
        """Compute dice loss between sigmoid predictions and binary targets."""
        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.flatten(1)
        target_flat = target.flatten(1)
        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1)
        loss = 1 - (2 * intersection + 1) / (union + 1)
        return loss.mean()

    def sigmoid_bce_loss(self, pred, target):
        """Compute binary cross-entropy loss with logits."""
        return F.binary_cross_entropy_with_logits(pred, target, reduction="mean")

    def _best_of_multimask_loss(
        self,
        pred_masks: Tensor,
        gt_masks: Tensor,
    ) -> Tensor:
        """Per-sample best-of-multimask BCE + Dice, averaged over batch."""
        _, num_masks, mask_h, mask_w = pred_masks.shape
        gt_resized = F.interpolate(
            gt_masks.float().unsqueeze(1),
            size=(mask_h, mask_w),
            mode="nearest",
        )
        gt_expanded = gt_resized.expand(-1, num_masks, -1, -1)

        bce = F.binary_cross_entropy_with_logits(
            pred_masks, gt_expanded, reduction="none"
        ).mean(dim=(2, 3))

        pred_sigmoid = torch.sigmoid(pred_masks)
        pred_flat = pred_sigmoid.flatten(2)
        target_flat = gt_expanded.flatten(2)
        intersection = (pred_flat * target_flat).sum(2)
        union = pred_flat.sum(2) + target_flat.sum(2)
        dice = 1 - (2 * intersection + 1) / (union + 1)

        per_sample = (bce + dice).min(dim=1).values
        return per_sample.mean()

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        target_image: Float[Tensor, "batch view 3 height width"] | None = None,
    ) -> Float[Tensor, ""]:
        label_maps = batch["target"]["label"]
        target_view_count = label_maps.shape[1]
        rendered_features = prediction.feature[:, :target_view_count]
        B, V, _, _, _ = rendered_features.shape
        device = rendered_features.device

        feature_rows: list[Tensor] = []
        point_coord_rows: list[Tensor] = []
        point_label_rows: list[Tensor] = []
        gt_mask_rows: list[Tensor] = []

        for b in range(B):
            for v in range(V):
                label_map = label_maps[b, v]
                binary_masks = decompose_label_map(label_map).to(device)

                if binary_masks.shape[0] == 0:
                    logger.warning(
                        f"Skipping prompted loss for batch {b}, view {v}: "
                        "label map is all background"
                    )
                    continue

                valid_masks = [
                    i
                    for i in range(binary_masks.shape[0])
                    if binary_masks[i].sum().item() >= self.cfg.min_object_pixels
                ]
                if len(valid_masks) == 0:
                    logger.warning(
                        f"Skipping prompted loss for batch {b}, view {v}: "
                        "no mask with enough pixels"
                    )
                    continue

                point_coords, point_labels, gt_mask = self.prompt_sampler.sample(
                    binary_masks
                )
                feature_rows.append(rendered_features[b, v])
                point_coord_rows.append(point_coords)
                point_label_rows.append(point_labels)
                gt_mask_rows.append(gt_mask.to(device))

        if not feature_rows:
            return torch.tensor(0.0, device=device, requires_grad=True)

        features_64 = F.interpolate(
            torch.stack(feature_rows, dim=0),
            size=(64, 64),
            mode="bilinear",
            align_corners=False,
        )
        point_coords_batch = torch.stack(point_coord_rows, dim=0).to(device)
        point_labels_batch = torch.stack(point_label_rows, dim=0).to(device)
        gt_masks_batch = torch.stack(gt_mask_rows, dim=0)

        pred_masks = self.mask_decoder(
            features_64,
            point_coords=point_coords_batch,
            point_labels=point_labels_batch,
        )
        loss = self._best_of_multimask_loss(pred_masks, gt_masks_batch)
        return self.cfg.weight * loss
