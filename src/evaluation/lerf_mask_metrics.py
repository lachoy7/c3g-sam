"""Backward-compatible re-exports; prefer src.evaluation.mask_metrics."""

from .mask_metrics import (
    MaskMetricScores,
    best_multimask_scores,
    boundary_iou,
    mask_iou,
    mask_to_boundary,
    mean_scores,
    scores_from_logits,
    warp_mask_iou,
    warp_mask_to_pose,
)

# Legacy names used by early LERF-Mask scripts
calculate_iou = mask_iou


def best_mask_iou(pred, gt):
    s = best_multimask_scores(pred, gt)
    return s.iou, s.boundary_iou, s.best_index
