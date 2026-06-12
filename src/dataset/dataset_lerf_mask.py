"""LERF-Mask benchmark loader (evaluation / test only).

Not for training. Use via ``+evaluation=lerf_mask`` with ``mode=test``, or
``python -m src.eval_lerf_mask``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchvision.transforms as tf
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..misc.cam_utils import camera_normalization
from .dataset import DatasetCfgCommon
from .lerf_mask_io import (
    LERF_MASK_SCENES,
    camera_to_extrinsics_intrinsics,
    list_mask_prompts,
    load_binary_mask,
    split_train_test_cameras,
)
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class LerfMaskCfg(DatasetCfgCommon):
    name: Literal["lerf_mask"]
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    scenes: list[str] = field(default_factory=lambda: list(LERF_MASK_SCENES))
    max_context_views: int = 24


@dataclass
class DatasetLerfMaskCfgWrapper:
    """Hydra wrapper; only ``lerf_mask_eval`` is supported (no training preset)."""

    lerf_mask_eval: LerfMaskCfg


class DatasetLerfMask(IterableDataset):
    """Eval-only: one (scene, test view, text prompt) with context train views."""

    _EVAL_ONLY_MSG = (
        "LERF-Mask is evaluation-only. Use mode=test with +evaluation=lerf_mask, "
        "or: python -m src.eval_lerf_mask"
    )

    cfg: LerfMaskCfg
    stage: Stage
    view_sampler: ViewSampler

    near: float = 0.01
    far: float = 100.0

    def __init__(
        self,
        cfg: LerfMaskCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        if stage != "test":
            raise ValueError(self._EVAL_ONLY_MSG)
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        root = Path(cfg.roots[0])
        missing = [s for s in cfg.scenes if not (root / s).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"LERF-Mask scenes not found under {root}: {missing}. "
                "Run scripts/download_lerf_mask.sh to fetch the dataset."
            )
        self.scene_list = list(cfg.scenes)

    def shuffle(self, items: list) -> list:
        indices = torch.randperm(len(items))
        return [items[i] for i in indices]

    def _subsample_context_indices(self, n: int) -> list[int]:
        if self.cfg.max_context_views <= 0 or n <= self.cfg.max_context_views:
            return list(range(n))
        idx = np.linspace(0, n - 1, self.cfg.max_context_views, dtype=int)
        return sorted(set(idx.tolist()))

    def get_bound(
        self, bound: Literal["near", "far"], num_views: int
    ) -> Tensor:
        value = torch.ones(num_views) * getattr(self, bound)
        return value

    def __iter__(self):
        if self.stage != "test":
            raise ValueError(self._EVAL_ONLY_MSG)
        worker_info = torch.utils.data.get_worker_info()
        scene_list = self.scene_list
        if worker_info is not None:
            scene_list = [
                s
                for i, s in enumerate(scene_list)
                if i % worker_info.num_workers == worker_info.id
            ]

        root = Path(self.cfg.roots[0])
        target_shape = tuple(self.cfg.input_image_shape)

        for scene_id in scene_list:
            scene_root = root / scene_id
            train_cams, test_cams = split_train_test_cameras(scene_root)
            ctx_indices = self._subsample_context_indices(len(train_cams))

            for test_idx, test_cam in enumerate(test_cams):
                mask_prompts = list_mask_prompts(scene_root, test_idx)
                if not mask_prompts:
                    continue

                for mask_prompt in mask_prompts:
                    mask_path = (
                        scene_root / "test_mask" / str(test_idx) / f"{mask_prompt}.png"
                    )
                    if not mask_path.is_file():
                        alt = mask_path.with_suffix(".jpg")
                        if alt.is_file():
                            mask_path = alt
                        else:
                            continue

                    extrinsics_list: list[np.ndarray] = []
                    intrinsics_list: list[np.ndarray] = []
                    images_list: list[Tensor] = []
                    original_images_list: list[Tensor] = []

                    for ci in ctx_indices:
                        ext, intr, pil = camera_to_extrinsics_intrinsics(
                            train_cams[ci], target_shape
                        )
                        original_images_list.append(
                            self.to_tensor(Image.open(train_cams[ci].image_path).convert("RGB"))
                        )
                        extrinsics_list.append(ext)
                        intrinsics_list.append(intr)
                        images_list.append(self.to_tensor(pil))

                    test_ext, test_intr, test_pil = camera_to_extrinsics_intrinsics(
                        test_cam, target_shape
                    )
                    gt_mask = load_binary_mask(mask_path, target_shape)
                    original_test = self.to_tensor(
                        Image.open(test_cam.image_path).convert("RGB")
                    )

                    extrinsics = torch.from_numpy(
                        np.stack(extrinsics_list + [test_ext], axis=0).astype(np.float32)
                    )
                    intrinsics = torch.from_numpy(
                        np.stack(intrinsics_list + [test_intr], axis=0).astype(np.float32)
                    )
                    images = torch.stack(images_list + [self.to_tensor(test_pil)], dim=0)
                    original_images = torch.stack(
                        original_images_list + [original_test], dim=0
                    )
                    masks = torch.from_numpy(gt_mask.astype(np.bool_))

                    n_ctx = len(ctx_indices)
                    context_extrinsics = extrinsics[:n_ctx]
                    if self.cfg.make_baseline_1:
                        a = context_extrinsics[0, :3, 3]
                        b = context_extrinsics[-1, :3, 3]
                        scale = (a - b).norm()
                        if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                            continue
                        extrinsics[:, :3, 3] /= scale
                    else:
                        scale = 1.0

                    if self.cfg.relative_pose:
                        extrinsics = camera_normalization(extrinsics[0:1], extrinsics)

                    overlap = torch.tensor([0.5], dtype=torch.float32)

                    yield {
                        "context": {
                            "extrinsics": extrinsics[:n_ctx],
                            "intrinsics": intrinsics[:n_ctx],
                            "image": images[:n_ctx],
                            "original_image": original_images[:n_ctx],
                            "near": self.get_bound("near", n_ctx) / scale,
                            "far": self.get_bound("far", n_ctx) / scale,
                            "index": torch.tensor(ctx_indices, dtype=torch.int64),
                            "overlap": overlap,
                        },
                        "target": {
                            "extrinsics": extrinsics[n_ctx:],
                            "intrinsics": intrinsics[n_ctx:],
                            "image": images[n_ctx:],
                            "original_image": original_images[n_ctx:],
                            "masks": masks.unsqueeze(0),
                            "near": self.get_bound("near", 1) / scale,
                            "far": self.get_bound("far", 1) / scale,
                            "index": torch.tensor([test_idx], dtype=torch.int64),
                        },
                        "scene": scene_id,
                        "mask_prompt": mask_prompt,
                    }
