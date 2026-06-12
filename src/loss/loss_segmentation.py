import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
from ..model.sam_decoder import SAMMaskDecoderWrapper


@dataclass
class LossSegmentationCfg:
    weight: float
    sam_checkpoint: str = ""
    sam_model_variant: str = "sam_vit_h"
    use_lora: bool = False
    lora_rank: int = 4


@dataclass
class LossSegmentationCfgWrapper:
    segmentation: LossSegmentationCfg


class LossSegmentation(Loss[LossSegmentationCfg, LossSegmentationCfgWrapper]):
    """BCE + Dice loss for SAM mask predictions."""

    def __init__(self, cfg: LossSegmentationCfgWrapper):
        super().__init__(cfg)

        self.mask_decoder = SAMMaskDecoderWrapper(
            sam_checkpoint=self.cfg.sam_checkpoint,
            model_variant=self.cfg.sam_model_variant,
            use_lora=self.cfg.use_lora,
            lora_rank=self.cfg.lora_rank,
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

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        target_image: Float[Tensor, "batch view 3 height width"] | None = None,
    ) -> Float[Tensor, ""]:
        gt_masks = batch["target"]["masks"]

        B, V, C, H, W = prediction.feature.shape
        features_flat = rearrange(prediction.feature, "b v c h w -> (b v) c h w")

        pred_masks = self.mask_decoder(features_flat)
        _, num_masks, MH, MW = pred_masks.shape

        gt_masks_flat = rearrange(gt_masks, "b v h w -> (b v) 1 h w").float()

        bce = self.sigmoid_bce_loss(pred_masks, gt_masks_flat)
        dice = self.dice_loss(pred_masks, gt_masks_flat)

        return self.cfg.weight * (bce + dice)
