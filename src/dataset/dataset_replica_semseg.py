"""Replica SemSeg dataset loader for prompted SAM training.

Expects the following directory structure::

    <root>/replica/<scene>/results/frame{id:06d}.jpg
    <root>/replica/<scene>/results/depth{id:06d}.png
    <root>/replica/<scene>/traj.txt
    <root>/replica/cam_params.json
    <root>/replica_label_maps/<scene>/semantic_{id:06d}.png
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as tf
from einops import repeat
from PIL import Image
from torch.utils.data import IterableDataset

from ..misc.cam_utils import camera_normalization
from .dataset import DatasetCfgCommon

logger = logging.getLogger(__name__)

SCENES = [
    "office0",
    "office1",
    "office2",
    "office3",
    "office4",
    "room0",
    "room1",
    "room2",
]


@dataclass
class ReplicaSemSegCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    scenes: list[str]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    num_of_inputs: int = 2
    prompt_strategy: str = "centroid"
    min_object_pixels: int = 16


@dataclass
class DatasetReplicaSemSegCfgWrapper:
    replica_semseg: ReplicaSemSegCfg


class DatasetReplicaSemSeg(IterableDataset):
    """Loads Replica SemSeg dataset with semantic labels for prompted training."""

    near: float = 0.01
    far: float = 100.0

    def __init__(self, cfg, stage, view_sampler):
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        root = Path(cfg.roots[0])
        if not root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")

        self.root = root
        self.intrinsics = self.load_intrinsics()
        self.trajectories = {scene: self.load_trajectory(scene) for scene in cfg.scenes}

    def load_intrinsics(self):
        """Read shared intrinsics from cam_params.json."""
        cam_params_path = self.root / "replica" / "cam_params.json"
        with open(cam_params_path, "r") as f:
            params = json.load(f)

        fx = float(params["camera"]["fx"])
        fy = float(params["camera"]["fy"])
        cx = float(params["camera"]["cx"])
        cy = float(params["camera"]["cy"])

        K = np.zeros((3, 3), dtype=np.float32)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        K[2, 2] = 1.0
        return K

    def load_trajectory(self, scene):
        """Parse traj.txt: 16 floats per line -> list of 4x4 matrices."""
        traj_path = self.root / "replica" / scene / "traj.txt"
        poses = []
        with open(traj_path, "r") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                values = line.split()
                if len(values) != 16:
                    raise ValueError(
                        f"Expected 16 values per line in {traj_path}, "
                        f"got {len(values)} at line {line_num}"
                    )
                mat = np.array([float(v) for v in values], dtype=np.float32).reshape(
                    4, 4
                )
                poses.append(mat)
        return poses

    def decompose_labels(self, label_map):
        """Label map -> (K, H, W) binary masks for non-background objects."""
        unique_ids = np.unique(label_map)
        non_bg_ids = unique_ids[unique_ids != 0]

        if len(non_bg_ids) == 0:
            h, w = label_map.shape
            return torch.zeros((0, h, w), dtype=torch.float32)

        masks = []
        for obj_id in non_bg_ids:
            mask = (label_map == obj_id).astype(np.float32)
            masks.append(mask)

        return torch.from_numpy(np.stack(masks, axis=0))

    def get_num_frames(self, scene):
        """Get number of frames available for a scene."""
        return len(self.trajectories[scene])

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        scene_list = list(self.cfg.scenes)

        if self.stage == "test" and worker_info is not None:
            scene_list = [
                scene
                for idx, scene in enumerate(scene_list)
                if idx % worker_info.num_workers == worker_info.id
            ]

        for scene in scene_list:
            num_frames = self.get_num_frames(scene)
            if num_frames < self.cfg.num_of_inputs + 1:
                continue

            context_indices, target_indices = self.sample_views(scene, num_frames)

            for target_idx in target_indices:
                idxs = list(context_indices) + [target_idx]

                extrinsics_list = []
                intrinsics_list = []
                images_list = []
                label_list = []
                valid = True

                for frame_id in idxs:
                    rgb_path = (
                        self.root
                        / "replica"
                        / scene
                        / "results"
                        / f"frame{frame_id:06d}.jpg"
                    )
                    label_path = (
                        self.root
                        / "replica_label_maps"
                        / scene
                        / f"semantic_{frame_id:06d}.png"
                    )

                    if not rgb_path.exists() or not label_path.exists():
                        logger.warning(f"Missing files for {scene} frame {frame_id}")
                        valid = False
                        break

                    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                    if rgb is None:
                        logger.warning(f"Could not read {rgb_path}, skipping")
                        valid = False
                        break
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

                    label_map = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
                    if label_map is None:
                        logger.warning(f"Could not read {label_path}, skipping")
                        valid = False
                        break

                    h_target, w_target = self.cfg.input_image_shape
                    rgb_resized = cv2.resize(
                        rgb, (w_target, h_target), interpolation=cv2.INTER_LINEAR
                    )
                    label_resized = cv2.resize(
                        label_map, (w_target, h_target), interpolation=cv2.INTER_NEAREST
                    )

                    intrinsics = self.intrinsics.copy()
                    orig_h, orig_w = self.cfg.original_image_shape
                    intrinsics[0, :] *= w_target / orig_w
                    intrinsics[1, :] *= h_target / orig_h

                    intrinsics[0, :] /= w_target
                    intrinsics[1, :] /= h_target

                    pose = self.trajectories[scene][frame_id]
                    if np.any(np.isinf(pose)) or np.any(np.isnan(pose)):
                        valid = False
                        break

                    extrinsics_list.append(pose)
                    intrinsics_list.append(intrinsics)
                    images_list.append(self.to_tensor(Image.fromarray(rgb_resized)))
                    label_list.append(
                        torch.from_numpy(label_resized.astype(np.int64)).unsqueeze(0)
                    )

                if not valid or len(extrinsics_list) < len(idxs):
                    continue

                extrinsics = torch.from_numpy(
                    np.stack(extrinsics_list, axis=0).astype(np.float32)
                )
                intrinsics = torch.from_numpy(
                    np.stack(intrinsics_list, axis=0).astype(np.float32)
                )
                images = torch.stack(images_list, dim=0)
                labels = torch.cat(label_list, dim=0)

                num_ctx = self.cfg.num_of_inputs
                context_extrinsics = extrinsics[:num_ctx]

                if self.cfg.make_baseline_1:
                    a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
                    scale = (a - b).norm()
                    if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                        continue
                    extrinsics[:, :3, 3] /= scale
                else:
                    scale = 1.0

                if self.cfg.relative_pose:
                    extrinsics = camera_normalization(extrinsics[0:1], extrinsics)

                yield {
                    "context": {
                        "extrinsics": extrinsics[:num_ctx],
                        "intrinsics": intrinsics[:num_ctx],
                        "image": images[:num_ctx],
                        "label": labels[:num_ctx],
                        "near": self.get_bound("near", num_ctx) / scale,
                        "far": self.get_bound("far", num_ctx) / scale,
                        "index": torch.tensor(list(context_indices), dtype=torch.int64),
                        "overlap": 0,
                    },
                    "target": {
                        "extrinsics": extrinsics[num_ctx:],
                        "intrinsics": intrinsics[num_ctx:],
                        "image": images[num_ctx:],
                        "label": labels[num_ctx:],
                        "near": self.get_bound("near", 1) / scale,
                        "far": self.get_bound("far", 1) / scale,
                        "index": torch.tensor([target_idx], dtype=torch.int64),
                    },
                    "scene": scene,
                }

    def sample_views(self, scene, num_frames):
        """Sample context and target view indices for a scene."""
        if self.stage == "test":
            context_indices = [0, min(self.cfg.num_of_inputs, num_frames - 1)]
            target_indices = list(range(num_frames))
        else:
            max_gap = min(num_frames - 1, 10)
            left = torch.randint(0, num_frames - max_gap, size=()).item()
            right = left + torch.randint(2, max_gap + 1, size=()).item()
            right = min(right, num_frames - 1)
            context_indices = [left, right]

            num_targets = min(num_frames - 2, 1)
            low = left + 1
            high = right
            if high <= low:
                target_indices = [left]
            else:
                target_indices = [
                    torch.randint(low, high, size=()).item() for _ in range(num_targets)
                ]

        return context_indices, target_indices

    def get_bound(self, bound, num_views):
        """Get near/far bound repeated for num_views."""
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self):
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage


DATASET_CLASS = DatasetReplicaSemSeg
DATASET_NAMES = ("replica_semseg",)
CFG_WRAPPERS = (DatasetReplicaSemSegCfgWrapper,)
