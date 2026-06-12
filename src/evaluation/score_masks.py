#!/usr/bin/env python3
"""Score exported SAM / C3G-SAM masks against GT test labels (local runner).

Dense predictions merge per-class PNGs with logit-aware overlap resolution using
``{class_id}_logits.npy`` when present.

Examples::

    python -m src.evaluation.score_masks --experiment c3gsam --wait
    python -m src.evaluation.score_masks --experiment sam \\
        --replica-root datasets/replica --scannet-root datasets/scannet \\
        --pred-root outputs/vanilla-sam
    python -m src.evaluation.score_masks --experiment c3gsam --smoke --dataset replica
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from src.evaluation.eval_common import (
    REPLICA_2DSEG_SCENES,
    SCANNET_2DSEG_TEST_SCENES,
    DatasetName,
    ExperimentName,
    expected_mask_class_ids,
    find_smoke_scene,
    resolve_experiment,
    resolve_local_dataset_root,
    resolve_local_pred_root,
)

DEFAULT_MIN_OBJECT_PIXELS = 16
DEFAULT_DILATION_RATIO = 0.02
DEFAULT_NUM_WORKERS = 8
SCORES_FILENAME = "scores.json"
SCENES_CSV_FILENAME = "scores_by_scene.csv"
DENSE_MASK_FILENAME = "dense_mask.png"

SCENE_CSV_FIELDS = [
    "experiment",
    "dataset",
    "split",
    "scene_id",
    "num_frames",
    "iou",
    "boundary_iou",
    "warp_iou",
    "num_scored_classes",
    "num_warp_pairs",
    "missing_predictions",
    "skipped_warp_pairs",
]


def _score_gt_frame_task(args: tuple) -> tuple[int, int, int, int, int, int]:
    import numpy as np
    from PIL import Image

    from src.misc.frame_layout import FramePaths

    (
        scene_dir_str,
        pred_scene_dir_str,
        frame_id,
        min_object_pixels,
        dilation_ratio,
    ) = args
    scene_dir = Path(scene_dir_str)
    pred_scene_dir = Path(pred_scene_dir_str)

    paths = FramePaths.from_frame_id(scene_dir, frame_id)
    if not paths.label.is_file():
        return 0, 0, 0, 0, 0, 0

    gt_dense = np.array(Image.open(paths.label))
    gt_classes = set(
        expected_mask_class_ids(paths.label, min_object_pixels=min_object_pixels)
    )
    frame_pred_dir = pred_scene_dir / frame_id
    pred_dense, missing = _build_dense_pred_mask(
        frame_pred_dir,
        gt_dense.shape[:2],
        sorted(gt_classes),
    )
    _save_dense_pred_mask(frame_pred_dir, pred_dense)
    counts = _accumulate_frame_global_iou_counts(
        pred_dense,
        gt_dense,
        gt_classes,
        dilation_ratio=dilation_ratio,
    )
    return (
        counts.intersection,
        counts.union,
        counts.boundary_intersection,
        counts.boundary_union,
        counts.num_gt_classes,
        missing,
    )


def _score_warp_pair_task(args: tuple) -> tuple[float | None, int]:
    import numpy as np
    from PIL import Image

    from src.evaluation.mask_metrics import warp_mask_iou
    from src.misc.frame_layout import FramePaths

    (
        dataset,
        scene_dir_str,
        pred_scene_dir_str,
        frame_a,
        frame_b,
        min_object_pixels,
    ) = args
    scene_dir = Path(scene_dir_str)
    pred_scene_dir = Path(pred_scene_dir_str)

    paths_a = FramePaths.from_frame_id(scene_dir, frame_a)
    paths_b = FramePaths.from_frame_id(scene_dir, frame_b)

    if not (
        paths_a.camera.is_file()
        and paths_b.camera.is_file()
        and paths_a.label.is_file()
        and paths_b.label.is_file()
    ):
        return None, 1

    label_a = np.array(Image.open(paths_a.label))
    image_size = tuple(label_a.shape[:2])
    depth_a = _load_frame_depth_meters(
        scene_dir, frame_a, dataset=dataset, image_size=image_size
    )
    depth_b = _load_frame_depth_meters(
        scene_dir, frame_b, dataset=dataset, image_size=image_size
    )
    if depth_a is None or depth_b is None:
        return None, 1

    ext_a, int_a = _load_camera_npz(paths_a.camera)
    ext_b, int_b = _load_camera_npz(paths_b.camera)

    gt_classes_a = set(
        expected_mask_class_ids(paths_a.label, min_object_pixels=min_object_pixels)
    )
    gt_classes_b = set(
        expected_mask_class_ids(paths_b.label, min_object_pixels=min_object_pixels)
    )
    pred_dense_a, _ = _build_dense_pred_mask(
        pred_scene_dir / frame_a,
        image_size,
        sorted(gt_classes_a),
    )
    pred_dense_b, _ = _build_dense_pred_mask(
        pred_scene_dir / frame_b,
        image_size,
        sorted(gt_classes_b),
    )
    shared_classes = _classes_in_dense_map(pred_dense_a) & _classes_in_dense_map(
        pred_dense_b
    )
    if not shared_classes:
        return None, 1

    pair_scores: list[float] = []
    for class_id in sorted(shared_classes):
        pred_a = (pred_dense_a == class_id).astype(np.uint8)
        pred_b = (pred_dense_b == class_id).astype(np.uint8)
        pair_scores.append(
            warp_mask_iou(
                pred_a,
                pred_b,
                ext_a,
                ext_b,
                int_a,
                int_b,
                image_size,
                depth=depth_a,
            )
        )
        pair_scores.append(
            warp_mask_iou(
                pred_b,
                pred_a,
                ext_b,
                ext_a,
                int_b,
                int_a,
                image_size,
                depth=depth_b,
            )
        )

    if not pair_scores:
        return None, 1
    return float(np.mean(pair_scores)), 0


TEST_SCENES: dict[DatasetName, list[str]] = {
    "replica": list(REPLICA_2DSEG_SCENES),
    "scannet": list(SCANNET_2DSEG_TEST_SCENES),
}


@dataclass
class GlobalIouCounts:
    """Accumulated pixel counts for global (micro) semantic IoU."""

    intersection: int = 0
    union: int = 0
    boundary_intersection: int = 0
    boundary_union: int = 0
    num_gt_classes: int = 0

    def add(
        self,
        *,
        intersection: int = 0,
        union: int = 0,
        boundary_intersection: int = 0,
        boundary_union: int = 0,
        num_gt_classes: int = 0,
    ) -> None:
        self.intersection += intersection
        self.union += union
        self.boundary_intersection += boundary_intersection
        self.boundary_union += boundary_union
        self.num_gt_classes += num_gt_classes

    def add_counts(self, other: GlobalIouCounts) -> None:
        self.add(
            intersection=other.intersection,
            union=other.union,
            boundary_intersection=other.boundary_intersection,
            boundary_union=other.boundary_union,
            num_gt_classes=other.num_gt_classes,
        )


@dataclass
class DatasetScores:
    iou: float
    boundary_iou: float
    warp_iou: float | None
    num_scored_classes: int
    num_warp_pairs: int
    missing_predictions: int
    skipped_warp_pairs: int


@dataclass
class SceneScores:
    dataset: str
    scene_id: str
    num_frames: int
    iou: float
    boundary_iou: float
    warp_iou: float | None
    num_scored_classes: int
    num_warp_pairs: int
    missing_predictions: int
    skipped_warp_pairs: int


def _global_iou_from_counts(intersection: int, union: int) -> float:
    if union == 0:
        return 0.0
    return float(intersection / union)


def _mean_or_zero(values: list[float]) -> float:
    import numpy as np

    return float(np.mean(values)) if values else 0.0


def _scene_scores_from_parts(
    *,
    dataset: str,
    scene_id: str,
    num_frames: int,
    gt_counts: GlobalIouCounts,
    warp_ious: list[float],
    missing_predictions: int,
    skipped_warp_pairs: int,
) -> SceneScores:
    warp_mean = _mean_or_zero(warp_ious) if warp_ious else None
    return SceneScores(
        dataset=dataset,
        scene_id=scene_id,
        num_frames=num_frames,
        iou=_global_iou_from_counts(gt_counts.intersection, gt_counts.union),
        boundary_iou=_global_iou_from_counts(
            gt_counts.boundary_intersection, gt_counts.boundary_union
        ),
        warp_iou=warp_mean,
        num_scored_classes=gt_counts.num_gt_classes,
        num_warp_pairs=len(warp_ious),
        missing_predictions=missing_predictions,
        skipped_warp_pairs=skipped_warp_pairs,
    )


def _load_camera_npz(camera_path: Path) -> tuple:
    import numpy as np

    metadata = np.load(camera_path)
    pose = metadata["camera_pose"].astype(np.float32)
    intrinsics = metadata["camera_intrinsics"].astype(np.float32)
    return pose, intrinsics


def _load_pred_mask(pred_path: Path):
    import numpy as np
    from PIL import Image

    if not pred_path.is_file():
        return None
    return np.array(Image.open(pred_path))


def _pred_mask_to_bool(mask):
    import numpy as np

    if mask.dtype == np.bool_:
        return mask
    return mask > 128


def _classes_in_dense_map(dense) -> set[int]:
    import numpy as np

    return {int(class_id) for class_id in np.unique(dense) if class_id != 0}


def _resize_logits_to_label_shape(logits, label_shape: tuple[int, int]):
    import cv2

    if tuple(logits.shape[:2]) == label_shape:
        return logits
    height, width = label_shape
    return cv2.resize(
        logits.astype("float32", copy=False),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def _build_dense_pred_mask(
    frame_pred_dir: Path,
    label_shape: tuple[int, int],
    class_ids: list[int],
) -> tuple:
    """Merge per-class binary PNGs into one dense label map.

    Overlapping pixels are assigned to the class with the highest saved logit
    (``{class_id}_logits.npy``). Classes only claim pixels inside their binary mask.
    Predictions without logits are skipped and counted as missing.
    """
    import numpy as np

    dense = np.zeros(label_shape, dtype=np.int32)
    best_logit = np.full(label_shape, -np.inf, dtype=np.float32)
    missing_predictions = 0
    for class_id in class_ids:
        binary = _load_pred_mask(frame_pred_dir / f"{class_id}.png")
        logits_path = frame_pred_dir / f"{class_id}_logits.npy"
        if binary is None or not logits_path.is_file():
            missing_predictions += 1
            continue
        mask = _pred_mask_to_bool(binary)
        logits = np.load(logits_path).astype(np.float32)
        if tuple(logits.shape[:2]) != label_shape:
            logits = _resize_logits_to_label_shape(logits, label_shape)
        update = mask & (logits > best_logit)
        dense[update] = class_id
        best_logit[update] = logits[update]
    return dense, missing_predictions


def _save_dense_pred_mask(frame_pred_dir: Path, dense) -> None:
    """Write merged per-frame prediction map next to per-class exports."""
    import numpy as np
    from PIL import Image

    frame_pred_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(dense.astype(np.uint16)).save(
        frame_pred_dir / DENSE_MASK_FILENAME
    )


def _accumulate_frame_global_iou_counts(
    pred_dense,
    gt_dense,
    gt_classes: set[int],
    *,
    dilation_ratio: float,
) -> GlobalIouCounts:
    """Sum IoU counts over every GT-present class in one frame."""
    from src.evaluation.mask_metrics import (
        binary_boundary_iou_counts,
        binary_mask_iou_counts,
    )

    counts = GlobalIouCounts()
    for class_id in sorted(gt_classes):
        pred_bin = pred_dense == class_id
        gt_bin = gt_dense == class_id
        inter, union = binary_mask_iou_counts(pred_bin, gt_bin)
        b_inter, b_union = binary_boundary_iou_counts(
            pred_bin, gt_bin, dilation_ratio=dilation_ratio
        )
        counts.add(
            intersection=inter,
            union=union,
            boundary_intersection=b_inter,
            boundary_union=b_union,
            num_gt_classes=1,
        )
    return counts


def _frame_depth_path(scene_dir: Path, frame_id: str) -> Path:
    return scene_dir / f"{frame_id}_depth.png"


def _depth_to_meters(raw, dataset: DatasetName):
    import numpy as np

    depth = raw.astype(np.float32)
    if dataset == "scannet":
        # Prepared ScanNet volume stores sens depth as uint16 millimeters.
        return depth / 1000.0
    # Prepared Replica volume stores renderer depth as uint16 millimeters.
    return depth / 1000.0


def _load_frame_depth_meters(
    scene_dir: Path,
    frame_id: str,
    *,
    dataset: DatasetName,
    image_size: tuple[int, int],
):
    import cv2

    depth_path = _frame_depth_path(scene_dir, frame_id)
    if not depth_path.is_file():
        return None

    raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 3:
        raw = raw[..., 0]

    depth_m = _depth_to_meters(raw, dataset)
    if tuple(depth_m.shape[:2]) != image_size:
        depth_m = cv2.resize(
            depth_m,
            (image_size[1], image_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    depth_m = depth_m.copy()
    depth_m[depth_m <= 0] = 0.0
    return depth_m



def _score_gt_masks(
    *,
    dataset_root: Path,
    pred_root: Path,
    scenes: list[str],
    min_object_pixels: int,
    dilation_ratio: float,
    num_workers: int,
) -> tuple[GlobalIouCounts, int]:
    from src.misc.frame_layout import list_frame_ids

    tasks: list[tuple] = []
    for scene_id in scenes:
        scene_dir = dataset_root / scene_id
        if not scene_dir.is_dir():
            continue
        pred_scene_dir = pred_root / scene_id
        for frame_id in list_frame_ids(scene_dir):
            tasks.append(
                (
                    str(scene_dir),
                    str(pred_scene_dir),
                    frame_id,
                    min_object_pixels,
                    dilation_ratio,
                )
            )

    totals = GlobalIouCounts()
    missing_predictions = 0

    if not tasks:
        return totals, missing_predictions

    workers = max(1, min(num_workers, len(tasks)))
    if workers == 1:
        results = map(_score_gt_frame_task, tasks)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(_score_gt_frame_task, tasks, chunksize=8)

    for inter, union, b_inter, b_union, num_gt, missing in results:
        totals.add(
            intersection=inter,
            union=union,
            boundary_intersection=b_inter,
            boundary_union=b_union,
            num_gt_classes=num_gt,
        )
        missing_predictions += missing

    return totals, missing_predictions



def _score_adjacent_warp_masks(
    *,
    dataset: DatasetName,
    dataset_root: Path,
    pred_root: Path,
    scenes: list[str],
    min_object_pixels: int,
    num_workers: int,
) -> tuple[list[float], int]:
    from src.misc.frame_layout import list_frame_ids

    tasks: list[tuple] = []
    for scene_id in scenes:
        scene_dir = dataset_root / scene_id
        if not scene_dir.is_dir():
            continue

        frame_ids = list_frame_ids(scene_dir)
        pred_scene_dir = pred_root / scene_id
        for frame_index in range(len(frame_ids) - 1):
            tasks.append(
                (
                    dataset,
                    str(scene_dir),
                    str(pred_scene_dir),
                    frame_ids[frame_index],
                    frame_ids[frame_index + 1],
                    min_object_pixels,
                )
            )

    warp_ious: list[float] = []
    skipped_warp_pairs = 0

    if not tasks:
        return warp_ious, skipped_warp_pairs

    workers = max(1, min(num_workers, len(tasks)))
    if workers == 1:
        results = map(_score_warp_pair_task, tasks)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(_score_warp_pair_task, tasks, chunksize=4)

    for warp_iou, skipped in results:
        if warp_iou is not None:
            warp_ious.append(warp_iou)
        skipped_warp_pairs += skipped

    return warp_ious, skipped_warp_pairs


def _score_dataset(
    *,
    dataset: DatasetName,
    dataset_root: Path,
    pred_root: Path,
    min_object_pixels: int,
    dilation_ratio: float,
    scenes: list[str] | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> tuple[DatasetScores, list[SceneScores]]:
    import numpy as np
    from tqdm.auto import tqdm

    from src.misc.frame_layout import list_frame_ids

    scene_ids = scenes if scenes is not None else TEST_SCENES[dataset]
    predictions = pred_root / dataset

    dataset_gt_counts = GlobalIouCounts()
    all_warp_ious: list[float] = []
    missing_predictions = 0
    skipped_warp_pairs = 0
    scene_rows: list[SceneScores] = []

    progress = tqdm(
        scene_ids,
        desc=f"scores/{dataset}",
        unit="scene",
        dynamic_ncols=True,
    )
    for scene_id in progress:
        scene_dir = dataset_root / scene_id
        if not scene_dir.is_dir():
            continue

        num_frames = len(list_frame_ids(scene_dir))
        progress.set_postfix_str(scene_id)
        scene_gt_counts, scene_missing_predictions = _score_gt_masks(
            dataset_root=dataset_root,
            pred_root=predictions,
            scenes=[scene_id],
            min_object_pixels=min_object_pixels,
            dilation_ratio=dilation_ratio,
            num_workers=num_workers,
        )
        warp_ious, scene_skipped_warp_pairs = _score_adjacent_warp_masks(
            dataset=dataset,
            dataset_root=dataset_root,
            pred_root=predictions,
            scenes=[scene_id],
            min_object_pixels=min_object_pixels,
            num_workers=num_workers,
        )

        scene_row = _scene_scores_from_parts(
            dataset=dataset,
            scene_id=scene_id,
            num_frames=num_frames,
            gt_counts=scene_gt_counts,
            warp_ious=warp_ious,
            missing_predictions=scene_missing_predictions,
            skipped_warp_pairs=scene_skipped_warp_pairs,
        )
        scene_rows.append(scene_row)

        dataset_gt_counts.add_counts(scene_gt_counts)
        all_warp_ious.extend(warp_ious)
        missing_predictions += scene_missing_predictions
        skipped_warp_pairs += scene_skipped_warp_pairs

        progress.set_postfix(
            scene=scene_id,
            gt_classes=dataset_gt_counts.num_gt_classes,
            warp_pairs=len(all_warp_ious),
            missing=missing_predictions,
        )

    dataset_scores = DatasetScores(
        iou=_global_iou_from_counts(
            dataset_gt_counts.intersection, dataset_gt_counts.union
        ),
        boundary_iou=_global_iou_from_counts(
            dataset_gt_counts.boundary_intersection,
            dataset_gt_counts.boundary_union,
        ),
        warp_iou=float(np.mean(all_warp_ious)) if all_warp_ious else None,
        num_scored_classes=dataset_gt_counts.num_gt_classes,
        num_warp_pairs=len(all_warp_ious),
        missing_predictions=missing_predictions,
        skipped_warp_pairs=skipped_warp_pairs,
    )
    return dataset_scores, scene_rows


def _write_scene_scores_csv(
    *,
    path: Path,
    experiment: ExperimentName,
    scene_rows: list[SceneScores],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCENE_CSV_FIELDS)
        writer.writeheader()
        for row in scene_rows:
            writer.writerow(
                {
                    "experiment": experiment,
                    "dataset": row.dataset,
                    "split": "test",
                    "scene_id": row.scene_id,
                    "num_frames": row.num_frames,
                    "iou": f"{row.iou:.6f}",
                    "boundary_iou": f"{row.boundary_iou:.6f}",
                    "warp_iou": (
                        f"{row.warp_iou:.6f}" if row.warp_iou is not None else ""
                    ),
                    "num_scored_classes": row.num_scored_classes,
                    "num_warp_pairs": row.num_warp_pairs,
                    "missing_predictions": row.missing_predictions,
                    "skipped_warp_pairs": row.skipped_warp_pairs,
                }
            )


def run_scoring(
    *,
    experiment: ExperimentName,
    replica_root: str,
    scannet_root: str,
    pred_root: Path,
    min_object_pixels: int,
    dilation_ratio: float,
    scenes: dict[DatasetName, list[str]] | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> dict:
    results: dict[str, dict] = {}
    scene_rows: list[SceneScores] = []

    for dataset, dataset_root in (
        ("replica", Path(replica_root)),
        ("scannet", Path(scannet_root)),
    ):
        dataset_scenes = None
        if scenes is not None:
            dataset_scenes = scenes[dataset]  # type: ignore[index]

        print(f"[scores/{experiment}/{dataset}/test] scoring with {num_workers} workers...")
        scores, dataset_scene_rows = _score_dataset(
            dataset=dataset,  # type: ignore[arg-type]
            dataset_root=dataset_root,
            pred_root=pred_root,
            min_object_pixels=min_object_pixels,
            dilation_ratio=dilation_ratio,
            scenes=dataset_scenes,
            num_workers=num_workers,
        )
        scene_rows.extend(dataset_scene_rows)
        print(
            f"[scores/{experiment}/{dataset}/test] "
            f"iou={scores.iou:.4f} "
            f"boundary_iou={scores.boundary_iou:.4f} "
            f"warp_iou={scores.warp_iou} "
            f"classes={scores.num_scored_classes} "
            f"missing={scores.missing_predictions} "
            f"scenes={len(dataset_scene_rows)}"
        )
        results[dataset] = asdict(scores)

    csv_path = pred_root / SCENES_CSV_FILENAME
    _write_scene_scores_csv(
        path=csv_path,
        experiment=experiment,
        scene_rows=scene_rows,
    )
    print(f"Wrote per-scene CSV to {csv_path} ({len(scene_rows)} rows)")

    return {
        "experiment": experiment,
        "split": "test",
        "pred_root": str(pred_root),
        "min_object_pixels": min_object_pixels,
        "dilation_ratio": dilation_ratio,
        "num_workers": num_workers,
        "scores_json": str(pred_root / SCORES_FILENAME),
        "scores_csv": str(csv_path),
        "results": results,
    }


def run_smoke_scoring(
    *,
    experiment: str,
    dataset: str = "replica",
    dataset_root: str | None = None,
    pred_root: str | Path | None = None,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
    num_workers: int = 8,
) -> dict:
    """Score one test scene as a quick sanity check."""
    experiment_name = resolve_experiment(experiment)
    if dataset not in TEST_SCENES:
        raise ValueError(f"Unknown dataset {dataset!r}")

    data_root = resolve_local_dataset_root(dataset, dataset_root)  # type: ignore[arg-type]
    predictions_root = resolve_local_pred_root(experiment, pred_root)
    scene_id = find_smoke_scene(data_root, scenes=TEST_SCENES[dataset])  # type: ignore[index]

    scores, _ = _score_dataset(
        dataset=dataset,  # type: ignore[arg-type]
        dataset_root=Path(data_root),
        pred_root=predictions_root,
        min_object_pixels=min_object_pixels,
        dilation_ratio=DEFAULT_DILATION_RATIO,
        scenes=[scene_id],
        num_workers=num_workers,
    )

    payload = {
        "experiment": experiment_name,
        "dataset": dataset,
        "split": "test",
        "scene_id": scene_id,
        "num_workers": num_workers,
        "scores": asdict(scores),
    }
    print(json.dumps(payload, indent=2))
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score exported SAM / C3G-SAM masks.")
    parser.add_argument(
        "--experiment",
        default="sam",
        help="Experiment id: sam, c3gsam, c3gsam_ema-mag-uproj, c3gsam_ema, c3gsam_noema-nomag",
    )
    parser.add_argument("--replica-root", type=Path, default=None)
    parser.add_argument("--scannet-root", type=Path, default=None)
    parser.add_argument("--pred-root", type=Path, default=None)
    parser.add_argument(
        "--min-object-pixels", type=int, default=DEFAULT_MIN_OBJECT_PIXELS
    )
    parser.add_argument(
        "--dilation-ratio", type=float, default=DEFAULT_DILATION_RATIO
    )
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Score a single test scene instead of the full test split.",
    )
    parser.add_argument("--dataset", default="replica", choices=("replica", "scannet"))
    args = parser.parse_args(argv)

    resolve_experiment(args.experiment)

    if args.smoke:
        dataset_root = args.replica_root if args.dataset == "replica" else args.scannet_root
        run_smoke_scoring(
            experiment=args.experiment,
            dataset=args.dataset,
            dataset_root=str(dataset_root) if dataset_root else None,
            pred_root=args.pred_root,
            min_object_pixels=args.min_object_pixels,
            num_workers=args.num_workers,
        )
        return

    experiment_name = resolve_experiment(args.experiment)
    replica_data_root = resolve_local_dataset_root("replica", str(args.replica_root) if args.replica_root else None)
    scannet_data_root = resolve_local_dataset_root("scannet", str(args.scannet_root) if args.scannet_root else None)
    predictions_root = resolve_local_pred_root(args.experiment, args.pred_root)

    for dataset_name, root in (
        ("replica", replica_data_root),
        ("scannet", scannet_data_root),
    ):
        if not Path(root).is_dir():
            raise FileNotFoundError(
                f"{dataset_name} dataset not found at {root}."
            )
    if not predictions_root.is_dir():
        raise FileNotFoundError(
            f"Predictions not found at {predictions_root}. "
            f"Run mask export for experiment={experiment_name!r} first."
        )

    report = run_scoring(
        experiment=experiment_name,
        replica_root=replica_data_root,
        scannet_root=scannet_data_root,
        pred_root=predictions_root,
        min_object_pixels=args.min_object_pixels,
        dilation_ratio=args.dilation_ratio,
        num_workers=args.num_workers,
    )

    out_path = args.output_path or predictions_root / SCORES_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote scores to {out_path}")


if __name__ == "__main__":
    main()
