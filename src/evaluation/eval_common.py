"""Shared mask eval constants and helpers (local + Modal)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from src.misc.frame_layout import FramePaths

# Keep in sync with dataset_replica_2dseg / dataset_scannet_2dseg (avoid importing
# src.dataset here — that package pulls in torch at import time).
REPLICA_2DSEG_SCENES = [
    "office0",
    "office1",
    "office2",
    "office3",
    "office4",
    "room0",
    "room1",
    "room2",
]
# Train split: scene0000_00 … scene0646_00 (647 scenes). Val then test = last 160 on volume.
SCANNET_2DSEG_SCENES = [f"scene{i:04d}_00" for i in range(807 - 80 - 80)]
# Test split: scene0727_00 … scene0806_00 (80 scenes).
SCANNET_2DSEG_TEST_SCENES = [f"scene{i:04d}_00" for i in range(807 - 80, 807)]

DatasetName = Literal["replica", "scannet"]

LOCAL_DATASET_ROOTS: dict[DatasetName, str] = {
    "replica": "datasets/replica",
    "scannet": "datasets/scannet",
}

# Vanilla SAM eval runs all Replica scenes plus ScanNet test only.
VANILLA_EVAL_DATASETS: list[tuple[DatasetName, list[str]]] = [
    ("replica", REPLICA_2DSEG_SCENES),
    ("scannet", SCANNET_2DSEG_TEST_SCENES),
]

# C3G distillation mask export: output subdir, Hydra dataset group name, scene list.
C3G_EVAL_DATASETS: list[tuple[DatasetName, str, list[str]]] = [
    ("replica", "replica_distill", REPLICA_2DSEG_SCENES),
    ("scannet", "scannet_distill", SCANNET_2DSEG_TEST_SCENES),
]

ExperimentName = Literal[
    "sam",
    "c3gsam",
    "c3gsam_ema-mag-uproj",
    "c3gsam_ema",
    "c3gsam_noema-nomag",
]

LOCAL_EXPERIMENT_PRED_ROOTS: dict[ExperimentName, Path] = {
    "sam": Path("outputs/vanilla-sam"),
    "c3gsam": Path("outputs/c3g-sam-eval"),
    "c3gsam_ema-mag-uproj": Path("outputs/c3g-sam-dft-eval"),
    "c3gsam_ema": Path("outputs/c3g-sam-nomaghead-eval"),
    "c3gsam_noema-nomag": Path("outputs/c3g-sam-ema-nomag-eval"),
}


def resolve_experiment(experiment: str) -> ExperimentName:
    """Normalize and validate a mask-export / scoring experiment identifier."""
    normalized = experiment.strip().lower()
    if normalized not in LOCAL_EXPERIMENT_PRED_ROOTS:
        names = "', '".join(LOCAL_EXPERIMENT_PRED_ROOTS)
        raise ValueError(
            f"Unknown experiment {experiment!r}; expected one of '{names}'."
        )
    return normalized  # type: ignore[return-value]


def resolve_local_dataset_root(
    dataset: DatasetName, dataset_root: str | None = None
) -> str:
    if dataset_root:
        return dataset_root
    env_key = f"C3G_{dataset.upper()}_ROOT"
    env_val = os.environ.get(env_key)
    if env_val:
        return env_val
    return LOCAL_DATASET_ROOTS[dataset]


def resolve_local_pred_root(
    experiment: str, pred_root: str | Path | None = None
) -> Path:
    experiment_name = resolve_experiment(experiment)
    if pred_root is not None:
        return Path(pred_root)
    env_key = f"C3G_PRED_{experiment_name.upper().replace('-', '_')}_ROOT"
    env_val = os.environ.get(env_key)
    if env_val:
        return Path(env_val)
    return LOCAL_EXPERIMENT_PRED_ROOTS[experiment_name]


def sample_eval_visualization_keys(
    dataset_root: str | Path,
    scenes: list[str],
    *,
    count: int = 5,
    seed: int = 42,
) -> list[str]:
    """Pick ``count`` random ``scene/frame_id`` keys for test debug visualizations."""
    import random

    frames = iter_dataset_frames(dataset_root, scenes)
    if not frames:
        raise FileNotFoundError(
            f"No labeled frames under {dataset_root} for scenes {scenes}"
        )
    rng = random.Random(seed)
    picked = rng.sample(frames, min(count, len(frames)))
    return [f"{scene}/{paths.frame_id}" for scene, paths in picked]


def find_smoke_scene(
    dataset_root: str | Path,
    *,
    scenes: list[str] | None = None,
) -> str:
    """Return the first scene that has prepared frames on disk."""
    from src.misc.frame_layout import list_frame_ids

    root = Path(dataset_root)
    index_path = root / "selected_seqs_test.json"
    if index_path.is_file():
        with index_path.open("r") as file_handle:
            indexed = json.load(file_handle)
        for scene_id in indexed:
            if list_frame_ids(root / scene_id):
                return scene_id

    for scene_id in scenes or []:
        if list_frame_ids(root / scene_id):
            return scene_id

    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir() or scene_dir.name.startswith(("_", ".")):
            continue
        if list_frame_ids(scene_dir):
            return scene_dir.name

    raise FileNotFoundError(
        f"No scene with frames found under {root}. "
        "Run the dataset download script to populate the dataset."
    )


def find_smoke_frame(
    dataset_root: str | Path,
    *,
    scenes: list[str] | None = None,
) -> tuple[str, FramePaths]:
    """Return the first scene and frame triplet available on disk."""
    from src.misc.frame_layout import FramePaths, list_frame_ids

    root = Path(dataset_root)
    scene_id = find_smoke_scene(root, scenes=scenes)
    scene_dir = root / scene_id
    frame_ids = list_frame_ids(scene_dir)
    if not frame_ids:
        raise FileNotFoundError(f"No frames under {scene_dir}")
    frame_id = frame_ids[0]
    paths = FramePaths.from_frame_id(scene_dir, frame_id)
    for path in (paths.image, paths.camera, paths.label):
        if not path.is_file():
            raise FileNotFoundError(f"Missing smoke-test frame file: {path}")
    return scene_id, paths


def iter_dataset_frames(
    dataset_root: str | Path,
    scenes: list[str],
    *,
    require_frames: bool = True,
) -> list[tuple[str, FramePaths]]:
    """List every (scene_id, frame paths) with image + label on disk."""
    from src.misc.frame_layout import FramePaths, list_frame_ids

    root = Path(dataset_root)
    frames: list[tuple[str, FramePaths]] = []
    for scene_id in scenes:
        scene_dir = root / scene_id
        if not scene_dir.is_dir():
            continue
        for frame_id in list_frame_ids(scene_dir):
            paths = FramePaths.from_frame_id(scene_dir, frame_id)
            if paths.image.is_file() and paths.label.is_file():
                frames.append((scene_id, paths))
    if require_frames and not frames:
        raise FileNotFoundError(
            f"No labeled frames found under {root} for scenes {scenes}. "
            "Run the dataset download script to populate the dataset."
        )
    return frames


def expected_mask_class_ids(
    label_path: Path,
    *,
    min_object_pixels: int = 16,
) -> list[int]:
    """Class ids that would receive a mask PNG (matches eval export skips)."""
    import numpy as np
    from PIL import Image

    label_np = np.array(Image.open(label_path))
    class_ids: list[int] = []
    for obj_id in np.unique(label_np):
        if obj_id == 0:
            continue
        if (label_np == obj_id).sum() < min_object_pixels:
            continue
        class_ids.append(int(obj_id))
    return class_ids


def frame_mask_export_complete(
    pred_root: Path,
    scene_id: str,
    frame_id: str,
    class_ids: list[int],
) -> bool:
    """True when every expected class mask PNG and logits NPY exist for this frame."""
    if not class_ids:
        return True
    frame_dir = pred_root / scene_id / frame_id
    return all(
        (frame_dir / f"{class_id}.png").is_file()
        and (frame_dir / f"{class_id}_logits.npy").is_file()
        for class_id in class_ids
    )


def scene_mask_export_complete(
    dataset_root: Path,
    scene_id: str,
    pred_root: Path,
    *,
    min_object_pixels: int = 16,
) -> bool:
    """True when all labeled frames in ``scene_id`` already have exported masks."""
    from src.misc.frame_layout import FramePaths, list_frame_ids

    scene_dir = dataset_root / scene_id
    if not scene_dir.is_dir():
        return False

    has_labeled_frame = False
    for frame_id in list_frame_ids(scene_dir):
        paths = FramePaths.from_frame_id(scene_dir, frame_id)
        if not (paths.image.is_file() and paths.label.is_file()):
            continue
        has_labeled_frame = True
        class_ids = expected_mask_class_ids(
            paths.label, min_object_pixels=min_object_pixels
        )
        if not frame_mask_export_complete(
            pred_root, scene_id, frame_id, class_ids
        ):
            return False

    return has_labeled_frame


def filter_scenes_for_mask_export(
    dataset_root: str | Path,
    scenes: list[str],
    pred_root: str | Path,
    *,
    min_object_pixels: int = 16,
) -> tuple[list[str], list[str]]:
    """Return ``(scenes_to_run, scenes_already_on_volume)``."""
    root = Path(dataset_root)
    out = Path(pred_root)
    pending: list[str] = []
    skipped: list[str] = []
    for scene_id in scenes:
        if scene_mask_export_complete(
            root, scene_id, out, min_object_pixels=min_object_pixels
        ):
            skipped.append(scene_id)
        else:
            pending.append(scene_id)
    return pending, skipped
