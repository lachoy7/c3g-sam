#!/usr/bin/env python3
"""Download and prepare ScanNet scenes for :mod:`dataset_scannet_2dseg`.

Writes one directory per scene under the output root::

    <out>/<scene_id>/{frame_id}_x.jpg
    <out>/<scene_id>/{frame_id}_cam.npz
    <out>/<scene_id>/{frame_id}_y.png
    <out>/<scene_id>/{frame_id}_depth.png

Also writes ``scannetv2-labels.combined.tsv`` and ``selected_seqs_test.json``.

Run on Modal (default: 712-scene 2D benchmark train+test; skips scenes on volume)::

    modal run src/dataset/download_scannet.py --accept-tos

All 807 scenes with public ``_2d-label-filt.zip``::

    modal run src/dataset/download_scannet.py --accept-tos --split all

Disconnect the local client but keep the job running on Modal::

    modal run --detach src/dataset/download_scannet.py --accept-tos

Block until all scenes finish (prints per-scene results)::

    modal run src/dataset/download_scannet.py --accept-tos --wait

Re-upload all scenes::

    modal run src/dataset/download_scannet.py --accept-tos --force

Run locally::

    python -m src.dataset.download_scannet --out-dir ./datasets/scannet --accept-tos
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
import ssl
import struct
import sys
import tempfile
import urllib.request
import zipfile
import zlib
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

BASE_URL = "http://kaldir.vc.cit.tum.de/scannet/"
TOS_URL = BASE_URL + "ScanNet_TOS.pdf"
RELEASE = "v2/scans"
RELEASE_TASKS = "v2/tasks"
LABEL_MAP_FILE = "scannetv2-labels.combined.tsv"

# ScanNet has ~1513 scans total; this script only prepares scenes that ship
# ``_2d-label-filt.zip`` (2D semantic seg). On the public server that is every
# ``scene0000_00`` … ``scene0806_00`` (807 scenes). The 2D benchmark split is
# 697 train (scene0000_00–scene0696_00) + 15 test (scene0697_00–scene0711_00).
SCENES_TEST_2D: tuple[str, ...] = tuple(f"scene{i:04d}_00" for i in range(697, 712))
SCENES_TRAIN_2D: tuple[str, ...] = tuple(f"scene{i:04d}_00" for i in range(0, 697))
SCENES_BENCHMARK_2D: tuple[str, ...] = SCENES_TRAIN_2D + SCENES_TEST_2D
SCENES_ALL_2D: tuple[str, ...] = tuple(f"scene{i:04d}_00" for i in range(0, 807))

SCENE_SPLITS: dict[str, tuple[str, ...]] = {
    "test": SCENES_TEST_2D,
    "train": SCENES_TRAIN_2D,
    "benchmark": SCENES_BENCHMARK_2D,
    "all": SCENES_ALL_2D,
}
SCENES: tuple[str, ...] = SCENES_BENCHMARK_2D


def sens_uses_v1_release(scan_id: str) -> bool:
    """``.sens`` for scene0000_00–scene0706_00 is on v1; scene0707_00+ on v2."""
    return int(scan_id[5:9]) < 707

FRAME_SKIP = 4
IMAGE_SIZE = (480, 640)  # (height, width)
FRAME_ID_WIDTH = 5
VOLUME_NAME = "scannet"
VOLUME_MOUNT = Path("/scannet")
RAW_SUBDIR = "_raw"
MODAL_RAW_SCRATCH = Path("/tmp/scannet_raw")
SCENE_COMPLETE_MARKER = ".complete"
SCENE_PREPARE_LOCK = ".preparing"
# Stale lock threshold: another worker may reclaim after this many seconds.
STALE_LOCK_SECONDS = 2 * 60 * 60
SELECTED_SEQS_FILE = "selected_seqs_test.json"
# Modal: one scene per container; cap parallel commits (Modal recommends ~5).
MODAL_MAX_PARALLEL_SCENES = 5

COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


def format_frame_id(frame_index: int) -> str:
    return f"{frame_index:0{FRAME_ID_WIDTH}d}"


def resolve_scene_split(split: str) -> tuple[str, ...]:
    try:
        return SCENE_SPLITS[split]
    except KeyError:
        valid = ", ".join(sorted(SCENE_SPLITS))
        raise SystemExit(f"Unknown --split {split!r}; choose one of: {valid}") from None


# ---------------------------------------------------------------------------
# ScanNet .sens reader (Python 3 port of ScanNet/SensReader/python/SensorData.py)
# ---------------------------------------------------------------------------


class RGBDFrame:
    def load(self, file_handle) -> None:
        self.camera_to_world = np.asarray(
            struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.color_data = file_handle.read(self.color_size_bytes)
        self.depth_data = file_handle.read(self.depth_size_bytes)

    def decompress_depth(self, compression_type: str) -> bytes:
        if compression_type == "zlib_ushort":
            return zlib.decompress(self.depth_data)
        raise ValueError(f"Unsupported depth compression: {compression_type}")

    def decompress_color(self, compression_type: str) -> np.ndarray:
        if compression_type == "jpeg":
            return imageio.imread(self.color_data)
        raise ValueError(f"Unsupported color compression: {compression_type}")


class SensorData:
    def __init__(self, filename: str | os.PathLike[str]) -> None:
        self.version = 4
        self.load(filename)

    def load(self, filename: str | os.PathLike[str]) -> None:
        with open(filename, "rb") as file_handle:
            version = struct.unpack("I", file_handle.read(4))[0]
            if version != self.version:
                raise ValueError(f"Unsupported .sens version {version}")
            strlen = struct.unpack("Q", file_handle.read(8))[0]
            file_handle.read(strlen)
            self.intrinsic_color = np.asarray(
                struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            file_handle.read(16 * 4)  # extrinsic_color
            file_handle.read(16 * 4)  # intrinsic_depth
            file_handle.read(16 * 4)  # extrinsic_depth
            color_compression = struct.unpack("i", file_handle.read(4))[0]
            depth_compression = struct.unpack("i", file_handle.read(4))[0]
            self.color_compression_type = COMPRESSION_TYPE_COLOR[color_compression]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[depth_compression]
            self.color_width = struct.unpack("I", file_handle.read(4))[0]
            self.color_height = struct.unpack("I", file_handle.read(4))[0]
            self.depth_width = struct.unpack("I", file_handle.read(4))[0]
            self.depth_height = struct.unpack("I", file_handle.read(4))[0]
            file_handle.read(4)  # depth_shift (unused for export)
            num_frames = struct.unpack("Q", file_handle.read(8))[0]
            self.frames: list[RGBDFrame] = []
            for _ in range(num_frames):
                frame = RGBDFrame()
                frame.load(file_handle)
                self.frames.append(frame)

    def export_frames_strided(
        self,
        output_dir: str | os.PathLike[str],
        *,
        frame_skip: int = 1,
        image_size: tuple[int, int] | None = None,
    ) -> list[int]:
        output_dir = Path(output_dir)
        color_dir = output_dir / "color"
        pose_dir = output_dir / "pose"
        color_dir.mkdir(parents=True, exist_ok=True)
        pose_dir.mkdir(parents=True, exist_ok=True)

        frame_indices = list(range(0, len(self.frames), frame_skip))
        for frame_idx in frame_indices:
            frame = self.frames[frame_idx]
            color = frame.decompress_color(self.color_compression_type)
            if image_size is not None:
                color = cv2.resize(
                    color,
                    (image_size[1], image_size[0]),
                    interpolation=cv2.INTER_AREA,
                )
            imageio.imwrite(color_dir / f"{frame_idx}.jpg", color)
            np.savetxt(pose_dir / f"{frame_idx}.txt", frame.camera_to_world)
        return frame_indices


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def download_file(url: str, out_file: str | os.PathLike[str]) -> None:
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.is_file():
        print(f"Skipping existing file {out_file}")
        return
    print(f"Downloading {url} -> {out_file}")
    with tempfile.NamedTemporaryFile(dir=out_file.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(out_file)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url} -> {out_file}: {exc}") from None


def download_scan_file(
    scan_id: str,
    suffix: str,
    out_dir: str | os.PathLike[str],
    *,
    use_v1_sens: bool = False,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_release = "v1/scans" if use_v1_sens and suffix == ".sens" else RELEASE
    url = f"{BASE_URL}{scan_release}/{scan_id}/{scan_id}{suffix}"
    out_file = out_dir / f"{scan_id}{suffix}"
    download_file(url, out_file)
    return out_file


def download_label_map(out_dir: str | os.PathLike[str]) -> Path:
    url = f"{BASE_URL}{RELEASE_TASKS}/{LABEL_MAP_FILE}"
    out_path = Path(out_dir) / LABEL_MAP_FILE
    download_file(url, out_path)
    return out_path


def read_label_mapping(label_map_file: str | os.PathLike[str]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    with open(label_map_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            mapping[int(row["id"])] = int(row["nyu40id"])
    return mapping


def map_label_image(image: np.ndarray, label_mapping: dict[int, int]) -> np.ndarray:
    mapped = np.copy(image)
    for raw_id, nyu40_id in label_mapping.items():
        mapped[image == raw_id] = nyu40_id
    return mapped.astype(np.uint16)


def adjust_intrinsic(
    intrinsic: np.ndarray,
    original_size: tuple[int, int],
    output_size: tuple[int, int],
) -> np.ndarray:
    orig_h, orig_w = original_size
    out_h, out_w = output_size
    scaled = intrinsic.copy()
    scaled[0, 0] *= out_w / orig_w
    scaled[1, 1] *= out_h / orig_h
    scaled[0, 2] *= out_w / orig_w
    scaled[1, 2] *= out_h / orig_h
    return scaled


def extract_label_zip(zip_path: Path, scene_dir: Path) -> Path:
    label_dir = scene_dir / "label-filt"
    if label_dir.is_dir() and any(label_dir.glob("*.png")):
        return label_dir
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(scene_dir)
    if not label_dir.is_dir():
        for candidate in scene_dir.iterdir():
            if candidate.is_dir() and "label" in candidate.name.lower():
                label_dir = candidate
                break
    if not label_dir.is_dir():
        raise FileNotFoundError(f"No label-filt directory found under {scene_dir}")
    return label_dir


def resize_depth(depth: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(
        depth,
        (image_size[1], image_size[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def decompress_frame_depth(
    sensor: SensorData,
    frame_idx: int,
    *,
    image_size: tuple[int, int] | None = None,
) -> np.ndarray:
    frame = sensor.frames[frame_idx]
    depth_bytes = frame.decompress_depth(sensor.depth_compression_type)
    depth = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(
        sensor.depth_height, sensor.depth_width
    )
    if image_size is not None:
        depth = resize_depth(depth, image_size)
    return depth


def resize_label(
    label_path: Path,
    label_mapping: dict[int, int],
    image_size: tuple[int, int],
) -> np.ndarray:
    image = np.array(imageio.imread(label_path))
    image = cv2.resize(
        image,
        (image_size[1], image_size[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return map_label_image(image, label_mapping)


def build_scene(
    scan_id: str,
    raw_dir: Path,
    out_root: Path,
    label_map_file: Path,
    *,
    frame_skip: int = FRAME_SKIP,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> list[str]:
    scene_raw = raw_dir / scan_id
    sens_path = scene_raw / f"{scan_id}.sens"
    label_zip = scene_raw / f"{scan_id}_2d-label-filt.zip"
    use_v1_sens = sens_uses_v1_release(scan_id)

    if not sens_path.is_file():
        download_scan_file(scan_id, ".sens", scene_raw, use_v1_sens=use_v1_sens)
    if not label_zip.is_file():
        download_scan_file(scan_id, "_2d-label-filt.zip", scene_raw)

    work_dir = scene_raw / "extracted"
    sensor = SensorData(sens_path)
    frame_indices = sensor.export_frames_strided(
        work_dir,
        frame_skip=frame_skip,
        image_size=image_size,
    )
    label_filt_dir = extract_label_zip(label_zip, scene_raw)
    label_mapping = read_label_mapping(label_map_file)
    original_size = (sensor.color_height, sensor.color_width)
    intrinsic = adjust_intrinsic(
        sensor.intrinsic_color[:3, :3].astype(np.float32),
        original_size,
        image_size,
    )

    scene_out = out_root / scan_id
    scene_out.mkdir(parents=True, exist_ok=True)

    frame_names: list[str] = []
    for frame_idx in frame_indices:
        label_src = label_filt_dir / f"{frame_idx}.png"
        if not label_src.is_file():
            print(f"Warning: missing label for {scan_id} frame {frame_idx}, skipping")
            continue

        frame_name = format_frame_id(frame_idx)
        frame_names.append(frame_name)
        shutil.copy2(
            work_dir / "color" / f"{frame_idx}.jpg",
            scene_out / f"{frame_name}_x.jpg",
        )
        label = resize_label(label_src, label_mapping, image_size)
        imageio.imwrite(scene_out / f"{frame_name}_y.png", label)

        depth = decompress_frame_depth(sensor, frame_idx, image_size=image_size)
        imageio.imwrite(scene_out / f"{frame_name}_depth.png", depth)

        pose = np.loadtxt(work_dir / "pose" / f"{frame_idx}.txt", dtype=np.float32)
        np.savez(
            scene_out / f"{frame_name}_cam.npz",
            camera_pose=pose,
            camera_intrinsics=intrinsic,
        )

    return frame_names


def read_scene_complete_marker(scene_dir: Path) -> list[str] | None:
    marker = scene_dir / SCENE_COMPLETE_MARKER
    if not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text())
    except json.JSONDecodeError:
        return None
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        return None
    return [str(frame_id) for frame_id in frames]


def read_selected_seqs_manifest(out_root: Path) -> dict[str, list[str]]:
    manifest_path = out_root / SELECTED_SEQS_FILE
    if not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(scan_id): [str(frame_id) for frame_id in frame_ids]
        for scan_id, frame_ids in payload.items()
        if isinstance(frame_ids, list) and frame_ids
    }


def _frames_have_required_files(scene_dir: Path, frame_ids: list[str]) -> bool:
    if not frame_ids:
        return False
    return all(
        (scene_dir / f"{frame_id}_x.jpg").is_file()
        and (scene_dir / f"{frame_id}_y.png").is_file()
        and (scene_dir / f"{frame_id}_cam.npz").is_file()
        for frame_id in frame_ids
    )


def list_valid_frame_ids(scene_dir: Path) -> list[str]:
    """Frame ids with RGB, label, and camera files present (volume layout)."""
    if not scene_dir.is_dir():
        return []
    frame_ids: list[str] = []
    for image_path in scene_dir.glob("*_x.jpg"):
        if not image_path.is_file():
            continue
        frame_id = image_path.name[: -len("_x.jpg")]
        if _frames_have_required_files(scene_dir, [frame_id]):
            frame_ids.append(frame_id)
    return sorted(frame_ids)


def get_scene_frame_ids(
    scene_dir: Path,
    *,
    manifest: dict[str, list[str]] | None = None,
    scan_id: str | None = None,
) -> list[str] | None:
    """Return frame ids when the scene is already prepared on disk; else None."""
    if not scene_dir.is_dir():
        return None

    if manifest and scan_id and scan_id in manifest:
        expected = manifest[scan_id]
        if _frames_have_required_files(scene_dir, expected):
            return expected
        return None

    marker_frames = read_scene_complete_marker(scene_dir)
    if marker_frames and _frames_have_required_files(scene_dir, marker_frames):
        return marker_frames

    on_disk = list_valid_frame_ids(scene_dir)
    return on_disk if on_disk else None


def scene_is_complete(
    scene_dir: Path,
    *,
    manifest: dict[str, list[str]] | None = None,
    scan_id: str | None = None,
) -> bool:
    return get_scene_frame_ids(scene_dir, manifest=manifest, scan_id=scan_id) is not None


def backfill_scene_complete_marker(scene_dir: Path, frame_ids: list[str]) -> None:
    if frame_ids and not (scene_dir / SCENE_COMPLETE_MARKER).is_file():
        mark_scene_complete(scene_dir, frame_ids)


def mark_scene_complete(scene_dir: Path, frame_names: list[str]) -> None:
    marker = scene_dir / SCENE_COMPLETE_MARKER
    marker.write_text(json.dumps({"frames": frame_names}, indent=2))


def _lock_is_stale(lock_path: Path) -> bool:
    if not lock_path.is_file():
        return True
    return (time.time() - lock_path.stat().st_mtime) > STALE_LOCK_SECONDS


def try_claim_scene(out_root: Path, scan_id: str) -> bool:
    """Return True if this worker should prepare ``scan_id``."""
    scene_out = out_root / scan_id
    scene_out.mkdir(parents=True, exist_ok=True)
    lock_path = scene_out / SCENE_PREPARE_LOCK
    if lock_path.is_file() and not _lock_is_stale(lock_path):
        return False
    lock_path.unlink(missing_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as lock_file:
        lock_file.write(f"pid={os.getpid()}\n")
    return True


def release_scene_claim(scene_dir: Path) -> None:
    (scene_dir / SCENE_PREPARE_LOCK).unlink(missing_ok=True)


def write_selected_seqs_manifest(
    out_root: Path,
    *,
    scenes: tuple[str, ...] = SCENES,
) -> dict[str, list[str]]:
    selected_seqs: dict[str, list[str]] = {}
    for scan_id in scenes:
        scene_dir = out_root / scan_id
        frame_names = get_scene_frame_ids(scene_dir, scan_id=scan_id)
        if frame_names:
            backfill_scene_complete_marker(scene_dir, frame_names)
            selected_seqs[scan_id] = frame_names
    manifest_path = out_root / SELECTED_SEQS_FILE
    with open(manifest_path, "w") as file_handle:
        json.dump(selected_seqs, file_handle, indent=2)
    return selected_seqs


def prepare_scene_if_needed(
    scan_id: str,
    raw_dir: Path,
    out_root: Path,
    label_map_file: Path,
    *,
    frame_skip: int = FRAME_SKIP,
    image_size: tuple[int, int] = IMAGE_SIZE,
    force: bool = False,
    manifest: dict[str, list[str]] | None = None,
) -> tuple[str, list[str], str]:
    """Prepare one scene. Returns ``(scan_id, frame_names, status)``."""
    scene_out = out_root / scan_id
    if manifest is None:
        manifest = read_selected_seqs_manifest(out_root)

    if not force:
        frame_names = get_scene_frame_ids(scene_out, manifest=manifest, scan_id=scan_id)
        if frame_names:
            backfill_scene_complete_marker(scene_out, frame_names)
            return scan_id, frame_names, f"skipped (already on volume, {len(frame_names)} frames)"

    if not force and not try_claim_scene(out_root, scan_id):
        frame_names = get_scene_frame_ids(scene_out, manifest=manifest, scan_id=scan_id)
        if frame_names:
            backfill_scene_complete_marker(scene_out, frame_names)
            return scan_id, frame_names, f"skipped (completed by peer, {len(frame_names)} frames)"
        return scan_id, [], "skipped (in progress elsewhere)"

    try:
        print(f"Preparing {scan_id} ...")
        frame_names = build_scene(
            scan_id,
            raw_dir,
            out_root,
            label_map_file,
            frame_skip=frame_skip,
            image_size=image_size,
        )
        if frame_names:
            mark_scene_complete(scene_out, frame_names)
        status = f"uploaded ({len(frame_names)} frames)"
        print(f"  {scan_id}: {status}")
        return scan_id, frame_names, status
    finally:
        release_scene_claim(scene_out)


def prepare_scannet(
    out_root: str | os.PathLike[str],
    *,
    scenes: tuple[str, ...] = SCENES,
    accept_tos: bool = False,
    raw_dir: str | os.PathLike[str] | None = None,
    force: bool = False,
) -> None:
    if not accept_tos:
        print(
            "By continuing you confirm agreement to the ScanNet terms of use:\n"
            f"  {TOS_URL}\n"
            "Re-run with --accept-tos to proceed."
        )
        sys.exit(1)

    out_root = Path(out_root)
    raw_dir = Path(raw_dir) if raw_dir is not None else out_root / RAW_SUBDIR
    out_root.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    label_map_path = download_label_map(out_root)

    for scan_id in scenes:
        prepare_scene_if_needed(
            scan_id,
            raw_dir,
            out_root,
            label_map_path,
            force=force,
        )

    selected_seqs = write_selected_seqs_manifest(out_root, scenes=scenes)
    print(f"Done. {len(selected_seqs)} scenes under {out_root}")
    print(f"Point dataset configs at: dataset.scannet_2dseg.roots=[{out_root}]")


# ---------------------------------------------------------------------------
# Modal entrypoint
# ---------------------------------------------------------------------------

try:
    import modal

    app = modal.App("c3g-scannet-download")
    scannet_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("libgl1", "libglib2.0-0")
        .pip_install(
            "numpy==1.26.4",
            "opencv-python-headless==4.10.0.84",
            "imageio==2.37.0",
        )
    )

    @app.function(
        image=image,
        volumes={str(VOLUME_MOUNT): scannet_volume},
        timeout=60 * 60 * 6,
        cpu=4,
        memory=32768,
        max_containers=MODAL_MAX_PARALLEL_SCENES,
    )
    def prepare_one_scene(scan_id: str, *, force: bool = False) -> str:
        """Download and upload a single scene if it is not already on the volume."""
        scannet_volume.reload()
        scene_out = VOLUME_MOUNT / scan_id
        manifest = read_selected_seqs_manifest(VOLUME_MOUNT)
        if not force:
            frames = get_scene_frame_ids(scene_out, manifest=manifest, scan_id=scan_id)
            if frames:
                backfill_scene_complete_marker(scene_out, frames)
                scannet_volume.commit()
                return f"{scan_id}: skipped (already on volume, {len(frames)} frames)"

        if MODAL_RAW_SCRATCH.exists():
            shutil.rmtree(MODAL_RAW_SCRATCH)
        MODAL_RAW_SCRATCH.mkdir(parents=True, exist_ok=True)
        try:
            label_map_path = download_label_map(VOLUME_MOUNT)
            _, frame_names, status = prepare_scene_if_needed(
                scan_id,
                MODAL_RAW_SCRATCH,
                VOLUME_MOUNT,
                label_map_path,
                force=force,
                manifest=manifest,
            )
            if "skipped" not in status:
                scannet_volume.commit()
            return f"{scan_id}: {status}"
        finally:
            shutil.rmtree(MODAL_RAW_SCRATCH, ignore_errors=True)

    @app.function(
        image=image,
        volumes={str(VOLUME_MOUNT): scannet_volume},
        timeout=60 * 10,
        cpu=1,
    )
    def finalize_scannet_volume(scenes: tuple[str, ...]) -> str:
        """Rebuild ``selected_seqs_test.json`` and drop any stray raw cache on the volume."""
        scannet_volume.reload()
        raw_on_volume = VOLUME_MOUNT / RAW_SUBDIR
        if raw_on_volume.exists():
            shutil.rmtree(raw_on_volume)
        selected_seqs = write_selected_seqs_manifest(VOLUME_MOUNT, scenes=scenes)
        scannet_volume.commit()
        return f"manifest: {len(selected_seqs)} scenes"

    @app.function(
        image=image,
        volumes={str(VOLUME_MOUNT): scannet_volume},
        timeout=60 * 60 * 12,
        cpu=2,
        memory=8192,
    )
    def populate_scannet_volume(
        scenes: tuple[str, ...],
        *,
        force: bool = False,
    ) -> list[str]:
        """Prepare scenes in parallel (skipping those already on the volume)."""
        results = list(prepare_one_scene.map(scenes, kwargs={"force": force}))
        summary = finalize_scannet_volume.remote(scenes)
        results.append(summary)
        return results

    @app.local_entrypoint()
    def modal_main(
        accept_tos: bool = False,
        force: bool = False,
        wait: bool = False,
        split: str = "benchmark",
    ) -> None:
        if not accept_tos:
            print(
                "By continuing you confirm agreement to the ScanNet terms of use:\n"
                f"  {TOS_URL}\n"
                "Re-run with --accept-tos to proceed."
            )
            sys.exit(1)

        # Always spawn first: ``modal run --detach`` disconnects the local client and
        # does not set a ``detach`` entrypoint flag. Blocking .remote() here often
        # never reaches Modal before the CLI exits.
        scenes = resolve_scene_split(split)
        handle = populate_scannet_volume.spawn(scenes, force=force)
        print(f"Started ScanNet volume populate on Modal ({len(scenes)} scenes, split={split}).")
        print(f"  call_id: {handle.object_id}")
        print(f"  logs:    modal app logs {app.name}")
        print("  status:  modal app list")

        if wait:
            results = handle.get()
            print("Finished. Results:")
            for line in results:
                print(f"  {line}")

except ImportError:
    app = None  # type: ignore[assignment]
    modal_main = None  # type: ignore[assignment,misc]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare ScanNet scenes for 2D segmentation."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Local output directory.",
    )
    parser.add_argument(
        "--accept-tos",
        action="store_true",
        help="Confirm ScanNet terms of use and start downloading.",
    )
    parser.add_argument(
        "--modal",
        action="store_true",
        help="Populate the Modal volume via modal run (requires modal package).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite scenes even if already present.",
    )
    parser.add_argument(
        "--split",
        choices=tuple(SCENE_SPLITS),
        default="benchmark",
        help=(
            "Which 2D-labeled scenes to fetch: "
            "test (15), train (697), benchmark (712 train+test), all (807)."
        ),
    )
    args = parser.parse_args()
    scenes = resolve_scene_split(args.split)

    if args.modal:
        if app is None:
            print("Install modal (`pip install modal`) to use --modal.", file=sys.stderr)
            sys.exit(1)
        modal_main(  # type: ignore[misc]
            accept_tos=args.accept_tos,
            force=args.force,
            split=args.split,
        )
        return

    out_dir = args.out_dir or Path("datasets") / VOLUME_NAME
    prepare_scannet(
        out_dir,
        scenes=scenes,
        accept_tos=args.accept_tos,
        force=args.force,
    )


if __name__ == "__main__":
    main()
