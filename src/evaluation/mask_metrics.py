"""Generic binary mask metrics: IoU, boundary IoU, and warp IoU (pred vs pred)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor


@dataclass
class MaskMetricScores:
    iou: float
    boundary_iou: float
    best_index: int = 0
    warp_iou: Optional[float] = None


def _to_bool_numpy(mask: np.ndarray | Tensor) -> np.ndarray:
    if isinstance(mask, Tensor):
        mask = mask.detach().cpu().numpy()
    if mask.dtype == np.bool_:
        return mask
    return mask > 128 if mask.dtype != np.bool_ else mask


def mask_to_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """Boundary band of a binary mask (Gaussian-Grouping / boundary-IoU style)."""
    import cv2

    mask_u8 = (_to_bool_numpy(mask)).astype(np.uint8)
    h, w = mask_u8.shape
    img_diag = np.sqrt(h**2 + w**2)
    dilation = max(1, int(round(dilation_ratio * img_diag)))
    padded = cv2.copyMakeBorder(mask_u8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(padded, kernel, iterations=dilation)[1 : h + 1, 1 : w + 1]
    return mask_u8 - eroded


def binary_mask_iou_counts(
    pred: np.ndarray | Tensor,
    gt: np.ndarray | Tensor,
) -> tuple[int, int]:
    """Pixel intersection and union for binary masks (for global IoU aggregation)."""
    m1 = _to_bool_numpy(pred)
    m2 = _to_bool_numpy(gt)
    intersection = int(np.logical_and(m1, m2).sum())
    union = int(np.logical_or(m1, m2).sum())
    return intersection, union


def binary_boundary_iou_counts(
    pred: np.ndarray | Tensor,
    gt: np.ndarray | Tensor,
    dilation_ratio: float = 0.02,
) -> tuple[int, int]:
    """Boundary-band intersection and union for binary masks."""
    pred_b = mask_to_boundary(_to_bool_numpy(pred).astype(np.uint8), dilation_ratio)
    gt_b = mask_to_boundary(_to_bool_numpy(gt).astype(np.uint8), dilation_ratio)
    intersection = int(((pred_b > 0) & (gt_b > 0)).sum())
    union = int(((pred_b > 0) | (gt_b > 0)).sum())
    return intersection, union


def mask_iou(
    pred: np.ndarray | Tensor,
    gt: np.ndarray | Tensor,
) -> float:
    """Standard binary IoU between prediction and ground-truth (or reference) mask."""
    intersection, union = binary_mask_iou_counts(pred, gt)
    if union == 0:
        return 1.0
    return float(intersection / union)


def boundary_iou(
    pred: np.ndarray | Tensor,
    gt: np.ndarray | Tensor,
    dilation_ratio: float = 0.02,
) -> float:
    """IoU computed on boundary pixels only (pred vs label mask)."""
    intersection, union = binary_boundary_iou_counts(pred, gt, dilation_ratio)
    if union == 0:
        return 1.0
    return float(intersection / union)


def best_multimask_scores(
    pred_masks: np.ndarray | Tensor,
    gt_mask: np.ndarray | Tensor,
    *,
    dilation_ratio: float = 0.02,
) -> MaskMetricScores:
    """
    When the model returns K masks (e.g. SAM multimask), pick the one with highest IoU
    vs ground truth and report its IoU and boundary IoU.
    """
    if isinstance(pred_masks, Tensor):
        pred_masks = pred_masks.detach().cpu().numpy()
    if pred_masks.ndim == 2:
        pred_masks = pred_masks[None]

    gt_mask = _to_bool_numpy(gt_mask)
    best_iou, best_biou, best_k = 0.0, 0.0, 0
    for k in range(pred_masks.shape[0]):
        iou = mask_iou(pred_masks[k], gt_mask)
        biou = boundary_iou(pred_masks[k], gt_mask, dilation_ratio=dilation_ratio)
        if iou > best_iou:
            best_iou, best_biou, best_k = iou, biou, k
    return MaskMetricScores(iou=best_iou, boundary_iou=best_biou, best_index=best_k)


def warp_mask_to_pose(
    mask: np.ndarray | Tensor,
    src_extrinsics: np.ndarray | Tensor,
    dst_extrinsics: np.ndarray | Tensor,
    src_intrinsics: np.ndarray | Tensor,
    dst_intrinsics: np.ndarray | Tensor,
    image_size: tuple[int, int],
    depth: np.ndarray | Tensor | None = None,
) -> np.ndarray:
    """Warp a binary mask from the source camera into the destination camera frame via depth reprojection."""
    import torch

    def _to_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        return x.float()

    mask_t = _to_tensor(mask)
    device = mask_t.device
    src_ext = _to_tensor(src_extrinsics).to(device)
    dst_ext = _to_tensor(dst_extrinsics).to(device)
    src_K = _to_tensor(src_intrinsics).to(device)
    dst_K = _to_tensor(dst_intrinsics).to(device)

    H, W = image_size
    if mask_t.shape != (H, W):
        mask_t = torch.nn.functional.interpolate(
            mask_t.float().unsqueeze(0).unsqueeze(0), size=(H, W), mode="nearest"
        ).squeeze()

    if depth is None:
        depth_t = torch.ones(H, W, device=device)
    else:
        depth_t = _to_tensor(depth).to(device)
        if depth_t.shape != (H, W):
            depth_t = torch.nn.functional.interpolate(
                depth_t.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    u_coords = u_coords.float()
    v_coords = v_coords.float()

    fx_s, fy_s = src_K[0, 0], src_K[1, 1]
    cx_s, cy_s = src_K[0, 2], src_K[1, 2]

    if fx_s < 2.0:
        fx_s, cx_s = fx_s * W, cx_s * W
        fy_s, cy_s = fy_s * H, cy_s * H

    z = depth_t
    x = (u_coords - cx_s) * z / fx_s
    y = (v_coords - cy_s) * z / fy_s
    pts_cam = torch.stack([x, y, z, torch.ones_like(z)], dim=-1)

    src_c2w = src_ext
    pts_world = (src_c2w @ pts_cam.reshape(-1, 4).T).T[:, :3]

    dst_w2c = torch.linalg.inv(dst_ext)
    pts_dst_cam = (dst_w2c[:3, :3] @ pts_world.T).T + dst_w2c[:3, 3]

    dst_K_scaled = dst_K.clone()
    fx_d, fy_d = dst_K_scaled[0, 0], dst_K_scaled[1, 1]
    cx_d, cy_d = dst_K_scaled[0, 2], dst_K_scaled[1, 2]
    if fx_d < 2.0:
        fx_d, cx_d = fx_d * W, cx_d * W
        fy_d, cy_d = fy_d * H, cy_d * H

    z_dst = pts_dst_cam[:, 2]
    valid = z_dst > 1e-6
    u_dst = (fx_d * pts_dst_cam[:, 0] / z_dst + cx_d).long()
    v_dst = (fy_d * pts_dst_cam[:, 1] / z_dst + cy_d).long()

    in_bounds = valid & (u_dst >= 0) & (u_dst < W) & (v_dst >= 0) & (v_dst < H)
    mask_flat = mask_t.reshape(-1).bool()
    valid_src_depth = z.reshape(-1) > 1e-6
    fg_and_valid = mask_flat & in_bounds & valid_src_depth

    warped = torch.zeros(H, W, dtype=torch.uint8, device=device)
    warped[v_dst[fg_and_valid], u_dst[fg_and_valid]] = 255

    return warped.detach().cpu().numpy()


def warp_mask_iou(
    pred_mask: np.ndarray | Tensor,
    reference_pred_mask: np.ndarray | Tensor,
    src_extrinsics: np.ndarray | Tensor,
    dst_extrinsics: np.ndarray | Tensor,
    src_intrinsics: np.ndarray | Tensor,
    dst_intrinsics: np.ndarray | Tensor,
    image_size: tuple[int, int],
    depth: np.ndarray | Tensor | None = None,
) -> float:
    """
    IoU between a predicted mask warped into another camera and a reference prediction.

    Unlike boundary_iou, both operands are predictions (not GT).
    """
    warped = warp_mask_to_pose(
        pred_mask,
        src_extrinsics,
        dst_extrinsics,
        src_intrinsics,
        dst_intrinsics,
        image_size,
        depth=depth,
    )
    return mask_iou(warped, reference_pred_mask)


def scores_from_logits(
    pred_logits: Tensor,
    gt_masks: Tensor,
    *,
    threshold: float = 0.5,
    dilation_ratio: float = 0.02,
) -> list[MaskMetricScores]:
    """
    Decode (N, K, H, W) or (N, 1, H, W) logits vs (N, 1, H, W) GT; return per-sample scores.
    """
    if gt_masks.dim() == 5:
        gt_masks = gt_masks.squeeze(2)
    if gt_masks.dim() == 3:
        gt_masks = gt_masks.unsqueeze(1)

    pred = pred_logits.sigmoid() > threshold
    if pred.shape[-2:] != gt_masks.shape[-2:]:
        pred = (
            torch.nn.functional.interpolate(
                pred.float(),
                size=gt_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            > threshold
        )

    n = pred.shape[0]
    results: list[MaskMetricScores] = []
    for i in range(n):
        pred_i = pred[i].detach().cpu().numpy()
        gt_i = gt_masks[i, 0].detach().cpu().numpy()
        results.append(
            best_multimask_scores(pred_i, gt_i, dilation_ratio=dilation_ratio)
        )
    return results


def mean_scores(scores: list[MaskMetricScores]) -> dict[str, float]:
    if not scores:
        return {"iou": 0.0, "boundary_iou": 0.0}
    warp_vals = [s.warp_iou for s in scores if s.warp_iou is not None]
    out = {
        "iou": float(np.mean([s.iou for s in scores])),
        "boundary_iou": float(np.mean([s.boundary_iou for s in scores])),
    }
    if warp_vals:
        out["warp_iou"] = float(np.mean(warp_vals))
    return out
