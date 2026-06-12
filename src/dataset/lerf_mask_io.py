"""COLMAP I/O for the LERF-Mask *evaluation* benchmark (Gaussian-Grouping layout).

Train views are context only; metrics run on held-out test views in test_mask/.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .utils import (
    focal2fov,
    read_extrinsics_binary,
    read_intrinsics_binary,
    readColmapCameras,
)


@dataclass(frozen=True)
class LerfMaskCamera:
    """One registered view: RGB path, COLMAP pose, and intrinsics."""

    image_name: str
    image_path: Path
    width: int
    height: int
    R: np.ndarray  # 3x3, transposed COLMAP rotation (world-to-camera)
    T: np.ndarray  # 3, COLMAP translation
    fov_x: float
    fov_y: float


def _sorted_train_names(scene_root: Path) -> list[str]:
    train_dir = scene_root / "images_train"
    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Expected training images at {train_dir}. "
            "Download LERF-Mask from HuggingFace (mqye/Gaussian-Grouping/data/lerf_mask)."
        )
    names = sorted(os.listdir(train_dir))
    return [n.split(".")[0] for n in names if not n.startswith(".")]


def load_colmap_cameras(scene_root: Path, images_subdir: str = "images") -> list[LerfMaskCamera]:
    sparse = scene_root / "sparse" / "0"
    extr_path = sparse / "images.bin"
    intr_path = sparse / "cameras.bin"
    if not extr_path.is_file() or not intr_path.is_file():
        raise FileNotFoundError(
            f"COLMAP sparse reconstruction not found under {sparse}. "
            "Each LERF-Mask scene must contain sparse/0/images.bin and cameras.bin."
        )

    cam_extrinsics = read_extrinsics_binary(str(extr_path))
    cam_intrinsics = read_intrinsics_binary(str(intr_path))
    cam_infos = readColmapCameras(
        cam_extrinsics,
        cam_intrinsics,
        str(scene_root / images_subdir),
    )
    cam_infos = sorted(cam_infos, key=lambda c: c.image_name)

    cameras: list[LerfMaskCamera] = []
    for info in cam_infos:
        if info.image is None:
            continue
        cameras.append(
            LerfMaskCamera(
                image_name=info.image_name,
                image_path=Path(info.image_path),
                width=info.width,
                height=info.height,
                R=np.asarray(info.R, dtype=np.float32),
                T=np.asarray(info.T, dtype=np.float32),
                fov_x=float(info.FovX),
                fov_y=float(info.FovY),
            )
        )
    return cameras


def split_train_test_cameras(
    scene_root: Path,
) -> tuple[list[LerfMaskCamera], list[LerfMaskCamera]]:
    """
    Split views using images_train/ (Gaussian-Grouping LERF-Mask convention).

    Train = views listed in images_train/; test = all other COLMAP-registered views.
    Test view index i matches test_mask/{i}/ on disk.
    """
    train_names = set(_sorted_train_names(scene_root))
    all_cams = load_colmap_cameras(scene_root)
    train = [c for c in all_cams if c.image_name in train_names]
    test = [c for c in all_cams if c.image_name not in train_names]
    if not train:
        raise RuntimeError(f"No training views matched images_train/ in {scene_root}")
    if not test:
        raise RuntimeError(f"No test views found for {scene_root}")
    return train, test


def camera_to_extrinsics_intrinsics(
    cam: LerfMaskCamera,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, Image.Image]:
    """
    Build 4x4 extrinsics (COLMAP W2C) and normalized 3x3 intrinsics for one view.

    Returns (extrinsics, intrinsics, resized_rgb_pil).
    """
    from .cropping import (
        bbox_from_intrinsics_in_out,
        camera_matrix_of_crop,
        crop_image_depthmap,
        rescale_image_depthmap,
    )

    h_out, w_out = target_shape
    image = Image.open(cam.image_path).convert("RGB")
    depth_stub = np.ones((cam.height, cam.width), dtype=np.float32)

    focal_x = cam.width / (2 * np.tan(cam.fov_x / 2))
    focal_y = cam.height / (2 * np.tan(cam.fov_y / 2))
    intrinsics = np.array(
        [
            [focal_x, 0.0, cam.width / 2],
            [0.0, focal_y, cam.height / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    resolution = np.array([w_out, h_out])
    image, depth_stub, intrinsics = rescale_image_depthmap(
        image, depth_stub, intrinsics, resolution
    )
    intrinsics2 = camera_matrix_of_crop(
        intrinsics, image.size, (w_out, h_out), offset_factor=0.5
    )
    crop_bbox = bbox_from_intrinsics_in_out(intrinsics, intrinsics2, (w_out, h_out))
    image, _, intrinsics = crop_image_depthmap(
        image, depth_stub, intrinsics, crop_bbox
    )

    intrinsics[0, :] /= w_out
    intrinsics[1, :] /= h_out

    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[:3, :3] = cam.R
    extrinsics[:3, 3] = cam.T

    return extrinsics, intrinsics, image


def load_binary_mask(path: Path, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Load a grayscale mask PNG; return bool array (True = foreground)."""
    mask = np.array(Image.open(path).convert("L"))
    mask = mask > 128
    if target_shape is not None:
        h, w = target_shape
        mask = np.array(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (w, h), resample=Image.NEAREST
            )
        ) > 128
    return mask


def list_mask_prompts(scene_root: Path, test_view_idx: int) -> list[str]:
    """Return text prompts (mask stem names) for one test view folder."""
    view_dir = scene_root / "test_mask" / str(test_view_idx)
    if not view_dir.is_dir():
        return []
    prompts = []
    for fname in sorted(os.listdir(view_dir)):
        if fname.startswith("."):
            continue
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            prompts.append(Path(fname).stem)
    return prompts


LERF_MASK_SCENES = ("figurines", "ramen", "teatime")
