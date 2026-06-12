#!/usr/bin/env python3
"""Pre-compute SAM ViT-H encoder features for all training frames.

Iterates over scenes in a dataset (Replica or ScanNet), encodes each frame's
RGB image with the frozen SAM image encoder, and saves the resulting 256×64×64
feature map as a `{frame_id}_sam.pt` file alongside the existing frame data.

Usage::

    python scripts/precompute_sam_features.py \
        --dataset-root /data/replica \
        --output-root /data/precomputed/replica \
        --dataset replica \
        --sam-checkpoint pretrained_weights/sam_vit_h.pth \
        --batch-size 8

    python scripts/precompute_sam_features.py \
        --dataset-root /data/scannet \
        --dataset scannet \
        --scenes scene0000_00 scene0001_00 \
        --sam-checkpoint pretrained_weights/sam_vit_h.pth \
        --overwrite
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.misc.frame_layout import list_frame_ids
from src.model.sam.loader import load_sam
from src.model.sam.preprocess import preprocess_images

logger = logging.getLogger(__name__)

# Default Replica scenes
REPLICA_SCENES = [
    "office0",
    "office1",
    "office2",
    "office3",
    "office4",
    "room0",
    "room1",
    "room2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-compute SAM encoder features for training frames.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory of the dataset (e.g. /data/replica).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Directory to write {frame_id}_sam.pt files. Defaults to "
            "--dataset-root (same layout as images/cameras)."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["replica", "scannet"],
        required=True,
        help="Dataset type: 'replica' or 'scannet'.",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Scene names to process. If not provided, defaults to all Replica "
            "scenes (for --dataset replica) or all scene directories found "
            "under --dataset-root (for --dataset scannet)."
        ),
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        required=True,
        help="Path to the SAM checkpoint file.",
    )
    parser.add_argument(
        "--sam-model-variant",
        type=str,
        default="sam_vit_h",
        help="SAM model variant (default: sam_vit_h).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of frames to encode per batch (default: 8).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pt files if they already exist.",
    )
    return parser.parse_args()


def discover_scenes(
    dataset_root: Path, dataset: str, scenes: list[str] | None
) -> list[str]:
    """Determine which scenes to process."""
    if scenes is not None and len(scenes) > 0:
        return scenes

    if dataset == "replica":
        return REPLICA_SCENES

    # For ScanNet, discover all scene directories under the root
    scene_dirs = sorted(d.name for d in dataset_root.iterdir() if d.is_dir())
    if not scene_dirs:
        raise ValueError(f"No scene directories found under {dataset_root}")
    return scene_dirs


def load_frame_image(image_path: Path) -> torch.Tensor | None:
    """Load an RGB image as a float32 tensor in [0, 255] with shape (3, H, W).

    Returns None if the image cannot be read.
    """
    rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if rgb is None:
        return None
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    # Convert to (3, H, W) float32 in [0, 255]
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
    return tensor


@torch.no_grad()
def encode_batch(
    sam,
    images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Encode a batch of images through the SAM image encoder.

    Args:
        sam: Loaded SAM model (frozen).
        images: (B, 3, H, W) float32 tensor in [0, 255].
        device: Device to run encoding on.

    Returns:
        (B, 256, 64, 64) float32 tensor of SAM image embeddings.
    """
    images = images.to(device)
    images = F.interpolate(
        images, size=(1024, 1024), mode="bilinear", align_corners=False
    )
    preprocessed = preprocess_images(sam, images)
    embeddings = sam.image_encoder(preprocessed)
    return embeddings.cpu()


def process_scene(
    sam,
    input_scene_dir: Path,
    output_scene_dir: Path,
    device: torch.device,
    batch_size: int,
    overwrite: bool,
) -> tuple[int, int]:
    """Process all frames in a scene directory.

    Reads RGB frames from ``input_scene_dir`` and writes SAM features to
    ``output_scene_dir`` (may be the same path).

    Returns:
        (processed_count, skipped_count) tuple.
    """
    frame_ids = list_frame_ids(input_scene_dir)
    if not frame_ids:
        logger.warning(f"No frames found in {input_scene_dir}, skipping scene.")
        return 0, 0

    output_scene_dir.mkdir(parents=True, exist_ok=True)

    # Filter out frames that already have .pt files (unless overwrite)
    frames_to_process: list[str] = []
    for fid in frame_ids:
        output_path = output_scene_dir / f"{fid}_sam.pt"
        if output_path.exists() and not overwrite:
            continue
        frames_to_process.append(fid)

    if not frames_to_process:
        logger.info(
            f"Scene {input_scene_dir.name}: all {len(frame_ids)} frames already "
            f"have SAM features, skipping."
        )
        return 0, 0

    logger.info(
        f"Scene {input_scene_dir.name}: processing {len(frames_to_process)}/{len(frame_ids)} frames."
    )

    processed = 0
    skipped = 0

    for start in tqdm(
        range(0, len(frames_to_process), batch_size),
        desc=f"{input_scene_dir.name} batches",
        unit="batch",
    ):
        batch_frame_ids = frames_to_process[start : start + batch_size]
        batch_images: list[torch.Tensor] = []
        readable_frame_ids: list[str] = []

        for fid in batch_frame_ids:
            image_path = input_scene_dir / f"{fid}_x.jpg"
            image_tensor = load_frame_image(image_path)

            if image_tensor is None:
                logger.warning(f"Could not read {image_path}, skipping frame.")
                skipped += 1
                continue

            batch_images.append(image_tensor)
            readable_frame_ids.append(fid)

        if not batch_images:
            continue

        embeddings = encode_batch(sam, torch.stack(batch_images), device)
        for i, frame_id in enumerate(readable_frame_ids):
            output_path = output_scene_dir / f"{frame_id}_sam.pt"
            torch.save(embeddings[i].clone(), output_path)
            processed += 1

    logger.info(
        f"Scene {input_scene_dir.name}: saved {processed} features, skipped {skipped} frames."
    )
    return processed, skipped


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")

    output_root = args.output_root or args.dataset_root

    # Validate dataset root
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    # Discover scenes
    scenes = discover_scenes(args.dataset_root, args.dataset, args.scenes)
    logger.info(
        f"Dataset: {args.dataset}, scenes: {len(scenes)}, "
        f"input: {args.dataset_root}, output: {output_root}"
    )

    # Load SAM model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        f"Loading SAM model ({args.sam_model_variant}) from {args.sam_checkpoint}"
    )
    sam = load_sam(
        model_variant=args.sam_model_variant,
        checkpoint_path=args.sam_checkpoint,
        freeze=True,
    )
    sam = sam.to(device)
    sam.eval()
    logger.info(f"SAM model loaded on {device}")

    # Process each scene
    total_processed = 0
    total_skipped = 0

    for scene_name in scenes:
        input_scene_dir = args.dataset_root / scene_name
        if not input_scene_dir.is_dir():
            logger.warning(f"Scene directory not found: {input_scene_dir}, skipping.")
            continue

        output_scene_dir = output_root / scene_name
        processed, skipped = process_scene(
            sam=sam,
            input_scene_dir=input_scene_dir,
            output_scene_dir=output_scene_dir,
            device=device,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
        total_processed += processed
        total_skipped += skipped

    logger.info(
        f"Done. Total: {total_processed} features saved, {total_skipped} frames skipped."
    )


if __name__ == "__main__":
    main()
