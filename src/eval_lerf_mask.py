"""
LERF-Mask evaluation only (not used for training).

Dataset: https://huggingface.co/mqye/Gaussian-Grouping/tree/main/data/lerf_mask

Examples:
  # Vanilla SAM on GT RGB (segment-everything, best-mask IoU)
  python -m src.eval_lerf_mask \\
    --mode vanilla_sam \\
    --data_root datasets/lerf_mask \\
    --sam_checkpoint pretrained_weights/sam_vit_h.pth

  # C3G+SAM via trained checkpoint (full NVS + feature mask decode)
  python -m src.eval_lerf_mask \\
    --mode c3g_sam \\
    --checkpoint path/to/checkpoint.ckpt \\
    --sam_checkpoint pretrained_weights/sam_vit_h.pth
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset.lerf_mask_io import LERF_MASK_SCENES, list_mask_prompts, load_binary_mask
from src.evaluation.mask_metrics import best_multimask_scores
from src.model.sam import forward as sam_forward, load_sam


def _aggregate_scores(
    scores: dict[str, list[float]],
) -> dict[str, float]:
    per_class = {k: float(np.mean(v)) for k, v in scores.items() if v}
    overall = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return {"per_class": per_class, "overall": overall}


def _print_results(
    title: str,
    iou_result: dict,
    boundary_result: dict,
    warp_result: dict | None = None,
) -> None:
    print(f"\n=== {title} ===")
    print("Mean IoU per class:", iou_result["per_class"])
    print("Overall Mean IoU:", iou_result["overall"])
    print("Mean boundary mIoU per class:", boundary_result["per_class"])
    print("Overall boundary mIoU:", boundary_result["overall"])
    if warp_result is not None:
        print("Mean warp mIoU per class:", warp_result["per_class"])
        print("Overall warp mIoU:", warp_result["overall"])


@torch.no_grad()
def eval_vanilla_sam(
    data_root: Path,
    scenes: list[str],
    sam_checkpoint: str,
    sam_variant: str,
    output_dir: Path | None,
) -> None:
    """Run SAM on each test RGB; pick best of multimask outputs vs GT (same as C3G SAM head)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = load_sam(sam_variant, sam_checkpoint, freeze=True).to(device).eval()

    iou_scores: dict[str, list[float]] = defaultdict(list)
    boundary_scores: dict[str, list[float]] = defaultdict(list)
    warp_scores: dict[str, list[float]] = defaultdict(list)

    for scene in scenes:
        scene_root = data_root / scene
        test_mask_root = scene_root / "test_mask"
        if not test_mask_root.is_dir():
            print(f"Skipping {scene}: no test_mask/")
            continue

        from src.dataset.lerf_mask_io import split_train_test_cameras

        _, test_cams = split_train_test_cameras(scene_root)

        for test_idx in sorted(
            int(p.name) for p in test_mask_root.iterdir() if p.is_dir() and p.name.isdigit()
        ):
            if test_idx >= len(test_cams):
                continue
            cam = test_cams[test_idx]
            image = Image.open(cam.image_path).convert("RGB")
            rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().to(device)
            rgb = rgb.unsqueeze(0)

            result = sam_forward(
                sam,
                rgb,
                segment_everything=True,
                multimask_output=True,
            )
            pred_masks = result["masks"][0].detach().cpu().numpy()

            for prompt in list_mask_prompts(scene_root, test_idx):
                mask_path = test_mask_root / str(test_idx) / f"{prompt}.png"
                if not mask_path.is_file():
                    mask_path = mask_path.with_suffix(".jpg")
                if not mask_path.is_file():
                    continue
                gt = load_binary_mask(mask_path)
                scores = best_multimask_scores(pred_masks, gt)
                iou_scores[prompt].append(scores.iou)
                boundary_scores[prompt].append(scores.boundary_iou)
                # Warp mIoU (pred vs other pred): not run until warp_mask_to_pose is implemented.
                # See ModelWrapper.compute_warp_mask_miou and mask_metrics.warp_mask_iou.

                if output_dir is not None:
                    out_path = (
                        output_dir / scene / str(test_idx) / f"{prompt}.png"
                    )
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    best_mask = pred_masks[scores.best_index]
                    Image.fromarray((best_mask.astype(np.uint8) * 255)).save(out_path)

    warp_agg = _aggregate_scores(warp_scores) if warp_scores else None
    _print_results(
        "Vanilla SAM",
        _aggregate_scores(iou_scores),
        _aggregate_scores(boundary_scores),
        warp_agg,
    )


def eval_c3g_sam(
    data_root: Path,
    scenes: list[str],
    checkpoint: str,
    sam_checkpoint: str,
    sam_variant: str,
    output_dir: Path | None,
    config_overrides: list[str] | None,
) -> None:
    """Run C3G+SAM eval via src.main (+evaluation=lerf_mask)."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "src.main",
        "+evaluation=lerf_mask",
        "mode=test",
        f"checkpointing.load={checkpoint}",
        f"train.sam_checkpoint={sam_checkpoint}",
        f"train.sam_model_variant={sam_variant}",
        "train.reproj_model=sam",
        "test.compute_scores=false",
        "test.save_image=false",
        "test.save_compare=false",
        "test.eval_lerf_mask=true",
        f"dataset.lerf_mask_eval.roots=[{str(data_root.resolve())}]",
        f"dataset.lerf_mask_eval.scenes={list(scenes)}",
    ]
    if output_dir is not None:
        cmd.append(f"test.output_path={str(output_dir.resolve())}")
    if config_overrides:
        cmd.extend(config_overrides)

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="LERF-Mask segmentation evaluation")
    parser.add_argument(
        "--mode",
        choices=("vanilla_sam", "c3g_sam"),
        required=True,
        help="vanilla_sam: SAM on GT RGB; c3g_sam: C3G render + SAM mask decoder",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("datasets/lerf_mask"),
        help="Root containing figurines/, ramen/, teatime/",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=list(LERF_MASK_SCENES),
        choices=list(LERF_MASK_SCENES),
    )
    parser.add_argument("--sam_checkpoint", type=str, default="")
    parser.add_argument("--sam_variant", type=str, default="sam_vit_h")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="C3G checkpoint (required for c3g_sam)",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--hydra_override",
        action="append",
        default=[],
        help="Extra Hydra overrides for c3g_sam mode",
    )
    args = parser.parse_args()

    if args.mode == "vanilla_sam":
        if not args.sam_checkpoint:
            parser.error("--sam_checkpoint is required for vanilla_sam")
        eval_vanilla_sam(
            args.data_root,
            args.scenes,
            args.sam_checkpoint,
            args.sam_variant,
            args.output_dir,
        )
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required for c3g_sam")
        if not args.sam_checkpoint:
            parser.error("--sam_checkpoint is required for c3g_sam")
        eval_c3g_sam(
            args.data_root,
            args.scenes,
            args.checkpoint,
            args.sam_checkpoint,
            args.sam_variant,
            args.output_dir,
            args.hydra_override,
        )


if __name__ == "__main__":
    main()
