#!/usr/bin/env python3
"""Prepare Replica semantic-segmentation scenes for :mod:`dataset_replica_2dseg`.

Reads a local ``replica_semseg`` tree and writes one directory per scene under the
output root (same layout as :mod:`download_scannet`)::

    <out>/<scene>/{frame_id}_x.jpg
    <out>/<scene>/{frame_id}_cam.npz
    <out>/<scene>/{frame_id}_y.png
    <out>/<scene>/{frame_id}_depth.png

Also writes ``selected_seqs_test.json``.

Run on Modal (mounts local ``replica_semseg`` and populates the volume)::

    modal run src/dataset/download_replica.py --source /path/to/replica_semseg

Run detached::

    modal run --detach src/dataset/download_replica.py

Run locally::

    python src/dataset/download_replica.py --out-dir ./datasets/replica
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

SCENES: tuple[str, ...] = (
    "office0",
    "office1",
    "office2",
    "office3",
    "office4",
    "room0",
    "room1",
    "room2",
)
FRAME_SKIP = 4
FRAME_ID_WIDTH = 5

VOLUME_NAME = "replica"
VOLUME_MOUNT = Path("/replica")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = SCRIPT_DIR / "replica_data" / "replica_semseg"


def format_frame_id(frame_index: int) -> str:
    return f"{frame_index:0{FRAME_ID_WIDTH}d}"


def load_cam_params(source: Path) -> dict:
    with (source / "replica" / "cam_params.json").open("r") as file_handle:
        return json.load(file_handle)


def intrinsic_matrix(cam_params: dict) -> np.ndarray:
    camera = cam_params["camera"]
    intrinsic = np.eye(3, dtype=np.float32)
    intrinsic[0, 0] = camera["fx"]
    intrinsic[1, 1] = camera["fy"]
    intrinsic[0, 2] = camera["cx"]
    intrinsic[1, 2] = camera["cy"]
    return intrinsic


def load_trajectory(traj_path: Path) -> list[np.ndarray]:
    poses: list[np.ndarray] = []
    with traj_path.open("r") as file_handle:
        for line in file_handle:
            values = np.fromstring(line, sep=" ", dtype=np.float32)
            if values.size != 16:
                continue
            poses.append(values.reshape(4, 4))
    return poses


def strided_frame_indices(
    num_frames: int, *, frame_skip: int = FRAME_SKIP
) -> list[int]:
    return list(range(0, num_frames, frame_skip))


def copy_rgb(source_results: Path, dest: Path, frame_idx: int) -> None:
    src = source_results / f"frame{frame_idx:06d}.jpg"
    if not src.is_file():
        raise FileNotFoundError(f"Missing RGB frame {src}")
    shutil.copy2(src, dest)


def copy_label(source_labels: Path, dest: Path, frame_idx: int) -> None:
    src = source_labels / f"semantic_{frame_idx:06d}.png"
    if not src.is_file():
        raise FileNotFoundError(f"Missing semantic label {src}")
    shutil.copy2(src, dest)


def copy_depth(source_results: Path, dest: Path, frame_idx: int) -> None:
    src = source_results / f"depth{frame_idx:06d}.png"
    if not src.is_file():
        raise FileNotFoundError(f"Missing depth frame {src}")
    shutil.copy2(src, dest)


def build_scene(
    scene: str,
    source: Path,
    out_root: Path,
    intrinsic_3x3: np.ndarray,
) -> list[str]:
    scene_src = source / "replica" / scene
    results_dir = scene_src / "results"
    labels_dir = source / "replica_label_maps" / scene
    traj_path = scene_src / "traj.txt"
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Missing scene directory {results_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing label directory {labels_dir}")
    if not traj_path.is_file():
        raise FileNotFoundError(f"Missing trajectory {traj_path}")

    poses = load_trajectory(traj_path)
    frame_indices = strided_frame_indices(len(poses))

    scene_out = out_root / scene
    scene_out.mkdir(parents=True, exist_ok=True)

    frame_names: list[str] = []
    for frame_idx in frame_indices:
        frame_name = format_frame_id(frame_idx)
        frame_names.append(frame_name)
        copy_rgb(results_dir, scene_out / f"{frame_name}_x.jpg", frame_idx)
        copy_label(labels_dir, scene_out / f"{frame_name}_y.png", frame_idx)
        copy_depth(results_dir, scene_out / f"{frame_name}_depth.png", frame_idx)
        np.savez(
            scene_out / f"{frame_name}_cam.npz",
            camera_pose=poses[frame_idx].astype(np.float32),
            camera_intrinsics=intrinsic_3x3,
        )

    return frame_names


def prepare_replica(
    source: str | os.PathLike[str],
    out_root: str | os.PathLike[str],
    *,
    scenes: tuple[str, ...] = SCENES,
) -> None:
    source = Path(source)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cam_params = load_cam_params(source)
    intrinsic_3x3 = intrinsic_matrix(cam_params)

    selected_seqs: dict[str, list[str]] = {}
    for scene in scenes:
        print(f"Preparing {scene} ...")
        frame_names = build_scene(scene, source, out_root, intrinsic_3x3)
        if frame_names:
            selected_seqs[scene] = frame_names
        print(f"  {len(frame_names)} frames")

    with open(out_root / "selected_seqs_test.json", "w") as file_handle:
        json.dump(selected_seqs, file_handle, indent=2)

    print(f"Done. {len(selected_seqs)} scenes under {out_root}")
    print(f"Point dataset configs at: dataset.replica_2dseg.roots=[{out_root}]")


# ---------------------------------------------------------------------------
# Modal entrypoint
# ---------------------------------------------------------------------------

try:
    import modal

    app = modal.App("c3g-replica-download")
    base_image = (
        modal.Image.debian_slim()
        .pip_install("numpy")
        .add_local_dir(str(DEFAULT_SOURCE), remote_path="/source")
    )
    replica_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    @app.function(
        image=base_image,
        volumes={str(VOLUME_MOUNT): replica_volume},
        timeout=60 * 60 * 4,
        cpu=4,
        memory=16384,
    )
    def populate_replica_volume(source_dir: str) -> None:
        prepare_replica(source_dir, VOLUME_MOUNT)
        replica_volume.commit()

    @app.local_entrypoint()
    def modal_main(source: str | None = None, detach: bool = False) -> None:
        from src.misc.modal_run import dispatch_remote

        source_path = Path(source or DEFAULT_SOURCE).resolve()
        default_source = DEFAULT_SOURCE.resolve()
        if source_path != default_source:
            print(
                "This Modal entrypoint mounts the default replica_semseg directory. "
                f"Move/link your data to {DEFAULT_SOURCE} or run locally with --source {source_path}.",
                file=sys.stderr,
            )
            sys.exit(1)
        if not source_path.is_dir():
            print(f"Source directory not found: {source_path}", file=sys.stderr)
            sys.exit(1)
        dispatch_remote(
            populate_replica_volume,
            "/source",
            detach=detach,
            job_name="Replica volume populate",
            app_name=app.name,
        )

except ImportError:
    app = None  # type: ignore[assignment]
    modal_main = None  # type: ignore[assignment,misc]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Replica semseg scenes for 2D segmentation."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Path to replica_semseg directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Local output directory.",
    )
    parser.add_argument(
        "--modal",
        action="store_true",
        help="Populate the Modal volume via modal run (requires modal package).",
    )
    args = parser.parse_args()

    if args.modal:
        if modal_main is None:
            print("Install modal (`pip install modal`) to use --modal.", file=sys.stderr)
            sys.exit(1)
        modal_main(source=str(args.source))
        return

    if not args.source.is_dir():
        print(f"Source directory not found: {args.source}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir or Path("datasets") / VOLUME_NAME
    prepare_replica(args.source, out_dir)


if __name__ == "__main__":
    main()
