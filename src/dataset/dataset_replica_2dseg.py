"""Replica 2D semantic segmentation loader (flat per-scene layout).

Expects data prepared by :mod:`download_replica` on the Modal ``replica`` volume::

    <root>/<scene_id>/{frame_id}_x.jpg
    <root>/<scene_id>/{frame_id}_cam.npz
    <root>/<scene_id>/{frame_id}_y.png

Sampling, preprocessing, and batch layout match :mod:`dataset_replica_semseg`;
only on-disk paths and pose/intrinsic loading differ (per-frame ``_cam.npz``).
"""

from __future__ import annotations

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
from ..misc.frame_layout import FramePaths, list_frame_ids
from .dataset import DatasetCfgCommon
from .view_sampler import ViewSampler

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
class Replica2dSegCfg(DatasetCfgCommon):
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
class DatasetReplica2dSegCfgWrapper:
    replica_2dseg: Replica2dSegCfg


class DatasetReplica2dSeg(IterableDataset):
    """Loads Replica flat 2D-seg volume with the same logic as ``DatasetReplicaSemSeg``."""

    near: float = 0.01
    far: float = 100.0

    def __init__(self, cfg, stage, view_sampler: ViewSampler):
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
        self.frame_ids = {
            scene: list_frame_ids(self.root / scene) for scene in cfg.scenes
        }

    def load_intrinsics(self):
        """Read shared intrinsics from the first available frame camera file."""
        for scene in self.cfg.scenes:
            frame_ids = list_frame_ids(self.root / scene)
            if not frame_ids:
                continue
            camera_path = FramePaths.from_frame_id(
                self.root / scene, frame_ids[0]
            ).camera
            metadata = np.load(camera_path)
            return metadata["camera_intrinsics"].astype(np.float32)

        raise FileNotFoundError(
            f"No camera files found under {self.root} for scenes {self.cfg.scenes}"
        )

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
        return len(self.frame_ids[scene])

    def _scene_camera_tensors(
        self, scene: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load per-frame extrinsics and intrinsics for view sampling."""
        frame_ids = self.frame_ids[scene]
        h_target, w_target = self.cfg.input_image_shape
        orig_h, orig_w = self.cfg.original_image_shape
        extrinsics_list: list[np.ndarray] = []
        intrinsics_list: list[np.ndarray] = []

        for frame_id in frame_ids:
            paths = FramePaths.from_frame_id(self.root / scene, frame_id)
            if not paths.camera.is_file():
                raise FileNotFoundError(f"Missing camera file {paths.camera}")
            metadata = np.load(paths.camera)
            pose = metadata["camera_pose"].astype(np.float32)
            if np.any(np.isinf(pose)) or np.any(np.isnan(pose)):
                raise ValueError(f"Invalid pose in {paths.camera}")

            intrinsics = self.intrinsics.copy()
            intrinsics[0, :] *= w_target / orig_w
            intrinsics[1, :] *= h_target / orig_h
            intrinsics[0, :] /= w_target
            intrinsics[1, :] /= h_target

            extrinsics_list.append(pose)
            intrinsics_list.append(intrinsics)

        return (
            torch.from_numpy(np.stack(extrinsics_list, axis=0).astype(np.float32)),
            torch.from_numpy(np.stack(intrinsics_list, axis=0).astype(np.float32)),
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        scene_list = list(self.cfg.scenes)

        if self.stage == "test" and worker_info is not None:
            scene_list = [
                scene
                for idx, scene in enumerate(scene_list)
                if idx % worker_info.num_workers == worker_info.id
            ]

        if self.cfg.overfit_to_scene is not None:
            scene_list = [s for s in scene_list if s == self.cfg.overfit_to_scene]

        for scene in scene_list:
            frame_ids = self.frame_ids[scene]
            num_frames = len(frame_ids)
            if num_frames < self.view_sampler.num_context_views + 1:
                continue

            try:
                scene_extrinsics, scene_intrinsics = self._scene_camera_tensors(scene)
                context_indices, target_indices, overlap = self.view_sampler.sample(
                    scene,
                    scene_extrinsics,
                    scene_intrinsics,
                )
            except ValueError:
                continue

            context_indices = context_indices.tolist()
            target_indices = target_indices.tolist()

            for target_idx in target_indices:
                idxs = list(context_indices) + [target_idx]

                extrinsics_list = []
                intrinsics_list = []
                images_list = []
                label_list = []
                valid = True

                for view_index in idxs:
                    frame_id = frame_ids[view_index]
                    paths = FramePaths.from_frame_id(self.root / scene, frame_id)

                    if not paths.image.is_file() or not paths.label.is_file():
                        logger.warning(
                            f"Missing files for {scene} frame {frame_id}"
                        )
                        valid = False
                        break

                    rgb = cv2.imread(str(paths.image), cv2.IMREAD_COLOR)
                    if rgb is None:
                        logger.warning(f"Could not read {paths.image}, skipping")
                        valid = False
                        break
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

                    label_map = cv2.imread(str(paths.label), cv2.IMREAD_UNCHANGED)
                    if label_map is None:
                        logger.warning(f"Could not read {paths.label}, skipping")
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

                    if not paths.camera.is_file():
                        logger.warning(f"Missing camera file {paths.camera}, skipping")
                        valid = False
                        break

                    metadata = np.load(paths.camera)
                    pose = metadata["camera_pose"].astype(np.float32)
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

                num_ctx = self.view_sampler.num_context_views
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

                context_frame_ids = [int(frame_ids[i]) for i in context_indices]
                target_frame_id = int(frame_ids[target_idx])

                yield {
                    "context": {
                        "extrinsics": extrinsics[:num_ctx],
                        "intrinsics": intrinsics[:num_ctx],
                        "image": images[:num_ctx],
                        "label": labels[:num_ctx],
                        "near": self.get_bound("near", num_ctx) / scale,
                        "far": self.get_bound("far", num_ctx) / scale,
                        "index": torch.tensor(context_frame_ids, dtype=torch.int64),
                        "overlap": overlap,
                    },
                    "target": {
                        "extrinsics": extrinsics[num_ctx:],
                        "intrinsics": intrinsics[num_ctx:],
                        "image": images[num_ctx:],
                        "label": labels[num_ctx:],
                        "near": self.get_bound("near", 1) / scale,
                        "far": self.get_bound("far", 1) / scale,
                        "index": torch.tensor([target_frame_id], dtype=torch.int64),
                    },
                    "scene": scene,
                }

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


DATASET_CLASS = DatasetReplica2dSeg
DATASET_NAMES = ("replica_2dseg",)
CFG_WRAPPERS = (DatasetReplica2dSegCfgWrapper,)
