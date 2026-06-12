"""ScanNet distillation dataset loader (pre-computed SAM features).

Expects data prepared by :mod:`scripts.precompute_sam_features`::

    <root>/<scene_id>/{frame_id}_x.jpg
    <root>/<scene_id>/{frame_id}_cam.npz
    <root>/<scene_id>/{frame_id}_sam.pt

Reuses view sampling, scene splitting, camera loading, and image preprocessing
from :mod:`dataset_scannet_2dseg`.  Produces ``sam_features``
(V, 256, 64, 64) instead of ``label`` or ``sam_image``.
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

from ..global_cfg import get_cfg
from ..misc.cam_utils import camera_normalization
from ..misc.frame_layout import FramePaths, list_frame_ids
from .dataset import DatasetCfgCommon
from .scannet_2dseg_splits import discover_scene_ids, scenes_for_stage
from .view_sampler import ViewSampler

logger = logging.getLogger(__name__)


@dataclass
class ScannetDistillCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    scenes: list[str]
    val_scene_count: int
    test_scene_count: int
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    num_of_inputs: int = 2
    sam_features_root: Path | None = None


@dataclass
class DatasetScannetDistillCfgWrapper:
    scannet_distill: ScannetDistillCfg


class DatasetScannetDistill(IterableDataset):
    """ScanNet loader that yields pre-computed SAM features alongside images."""

    near: float = 0.01
    far: float = 100.0

    def __init__(self, cfg: ScannetDistillCfg, stage: str, view_sampler: ViewSampler):
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        root = Path(cfg.roots[0])
        if not root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")

        self.root = root

        if cfg.overfit_to_scene is not None:
            self.scenes = [cfg.overfit_to_scene]
        elif cfg.scenes:
            all_scenes = cfg.scenes
            self.scenes = scenes_for_stage(
                stage,
                scene_ids=all_scenes,
                num_val=cfg.val_scene_count,
                num_test=cfg.test_scene_count,
            )
        else:
            all_scenes = discover_scene_ids(root)
            self.scenes = scenes_for_stage(
                stage,
                scene_ids=all_scenes,
                num_val=cfg.val_scene_count,
                num_test=cfg.test_scene_count,
            )

        if not self.scenes:
            raise FileNotFoundError(
                f"No ScanNet scenes for stage={stage!r} under {root}"
            )

        self.intrinsics = self._load_intrinsics()
        self.frame_ids = {
            scene: list_frame_ids(self.root / scene) for scene in self.scenes
        }

    def _load_intrinsics(self) -> np.ndarray:
        """Read shared intrinsics from the first available frame camera file."""
        for scene in self.scenes:
            frame_ids = list_frame_ids(self.root / scene)
            if not frame_ids:
                continue
            camera_path = FramePaths.from_frame_id(
                self.root / scene, frame_ids[0]
            ).camera
            metadata = np.load(camera_path)
            return metadata["camera_intrinsics"].astype(np.float32)

        raise FileNotFoundError(
            f"No camera files found under {self.root} for scenes {self.scenes}"
        )

    def _sam_feature_path(self, scene: str, frame_id: str) -> Path:
        """Return the path to the pre-computed SAM feature file for a frame."""
        feature_root = (
            Path(self.cfg.sam_features_root)
            if self.cfg.sam_features_root is not None
            else self.root
        )
        return feature_root / scene / f"{frame_id}_sam.pt"

    def get_num_frames(self, scene: str) -> int:
        """Get number of frames available for a scene."""
        return len(self.frame_ids[scene])

    def _scene_camera_tensors(self, scene: str) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _build_example(
        self,
        scene: str,
        context_indices: list[int],
        target_indices: list[int],
        overlap: torch.Tensor,
    ) -> dict | None:
        frame_ids = self.frame_ids[scene]
        idxs = list(context_indices) + list(target_indices)
        num_ctx = len(context_indices)
        num_target_views = len(target_indices)

        extrinsics_list = []
        intrinsics_list = []
        images_list = []
        sam_features_list = []

        for view_index in idxs:
            frame_id = frame_ids[view_index]
            paths = FramePaths.from_frame_id(self.root / scene, frame_id)
            sam_path = self._sam_feature_path(scene, frame_id)

            if not paths.image.is_file():
                logger.warning(f"Missing image for {scene} frame {frame_id}")
                return None

            if not sam_path.is_file():
                logger.warning(
                    f"Missing SAM features for {scene} frame {frame_id}: {sam_path}"
                )
                return None

            rgb = cv2.imread(str(paths.image), cv2.IMREAD_COLOR)
            if rgb is None:
                logger.warning(f"Could not read {paths.image}, skipping")
                return None
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            h_target, w_target = self.cfg.input_image_shape
            rgb_resized = cv2.resize(
                rgb, (w_target, h_target), interpolation=cv2.INTER_LINEAR
            )

            sam_feat = torch.load(sam_path, map_location="cpu")
            if sam_feat.shape != (256, 64, 64):
                logger.warning(
                    f"Unexpected SAM feature shape {sam_feat.shape} for "
                    f"{scene} frame {frame_id}, expected (256, 64, 64)"
                )
                return None

            intrinsics = self.intrinsics.copy()
            orig_h, orig_w = self.cfg.original_image_shape
            intrinsics[0, :] *= w_target / orig_w
            intrinsics[1, :] *= h_target / orig_h
            intrinsics[0, :] /= w_target
            intrinsics[1, :] /= h_target

            if not paths.camera.is_file():
                logger.warning(f"Missing camera file {paths.camera}, skipping")
                return None

            metadata = np.load(paths.camera)
            pose = metadata["camera_pose"].astype(np.float32)
            if np.any(np.isinf(pose)) or np.any(np.isnan(pose)):
                return None

            extrinsics_list.append(pose)
            intrinsics_list.append(intrinsics)
            images_list.append(self.to_tensor(Image.fromarray(rgb_resized)))
            sam_features_list.append(sam_feat)

        extrinsics = torch.from_numpy(
            np.stack(extrinsics_list, axis=0).astype(np.float32)
        )
        intrinsics = torch.from_numpy(
            np.stack(intrinsics_list, axis=0).astype(np.float32)
        )
        images = torch.stack(images_list, dim=0)
        sam_features = torch.stack(sam_features_list, dim=0)

        context_extrinsics = extrinsics[:num_ctx]
        if self.cfg.make_baseline_1:
            a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
            scale = (a - b).norm()
            if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                return None
            extrinsics[:, :3, 3] /= scale
        else:
            scale = 1.0

        if self.cfg.relative_pose:
            extrinsics = camera_normalization(extrinsics[0:1], extrinsics)

        context_frame_ids = [int(frame_ids[i]) for i in context_indices]
        target_frame_ids = [int(frame_ids[i]) for i in target_indices]

        return {
            "context": {
                "extrinsics": extrinsics[:num_ctx],
                "intrinsics": intrinsics[:num_ctx],
                "image": images[:num_ctx],
                "sam_features": sam_features[:num_ctx],
                "near": self.get_bound("near", num_ctx) / scale,
                "far": self.get_bound("far", num_ctx) / scale,
                "index": torch.tensor(context_frame_ids, dtype=torch.int64),
                "overlap": overlap,
            },
            "target": {
                "extrinsics": extrinsics[num_ctx:],
                "intrinsics": intrinsics[num_ctx:],
                "image": images[num_ctx:],
                "sam_features": sam_features[num_ctx:],
                "near": self.get_bound("near", num_target_views) / scale,
                "far": self.get_bound("far", num_target_views) / scale,
                "index": torch.tensor(target_frame_ids, dtype=torch.int64),
            },
            "scene": scene,
        }

    def _visualization_view_indices(
        self, scene: str, frame_id: str
    ) -> tuple[list[int], list[int]] | None:
        frame_ids = self.frame_ids.get(scene, [])
        if frame_id not in frame_ids:
            return None

        target_index = frame_ids.index(frame_id)
        num_frames = len(frame_ids)
        num_ctx = self.view_sampler.num_context_views
        num_target = self.view_sampler.num_target_views
        if num_frames < num_ctx + 1:
            return None

        if num_ctx == 1:
            context_indices = [0]
        elif num_ctx == 2:
            context_indices = [0, num_frames - 1]
        else:
            mid = num_frames // 2
            context_indices = [0, mid, num_frames - 1]
            while len(context_indices) < num_ctx:
                context_indices.insert(1, max(1, mid // 2))
            context_indices = context_indices[:num_ctx]

        target_indices = [target_index]
        for offset in range(1, num_frames):
            for candidate in (target_index + offset, target_index - offset):
                if 0 <= candidate < num_frames and candidate not in target_indices:
                    target_indices.append(candidate)
                if len(target_indices) >= num_target:
                    break
            if len(target_indices) >= num_target:
                break

        return context_indices, target_indices

    def _build_visualization_batch(self, scene: str, frame_id: str) -> dict | None:
        indices = self._visualization_view_indices(scene, frame_id)
        if indices is None:
            return None
        context_indices, target_indices = indices
        overlap = torch.tensor([0.5], dtype=torch.float32)
        return self._build_example(scene, context_indices, target_indices, overlap)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        scene_list = list(self.scenes)

        if self.stage == "test" and worker_info is not None:
            scene_list = [
                scene
                for idx, scene in enumerate(scene_list)
                if idx % worker_info.num_workers == worker_info.id
            ]

        if self.cfg.overfit_to_scene is not None:
            scene_list = [s for s in scene_list if s == self.cfg.overfit_to_scene]

        if self.stage == "test" and (worker_info is None or worker_info.id == 0):
            viz_keys = get_cfg().get("eval", {}).get("visualization_keys") or []
            for key in viz_keys:
                scene, frame_id = str(key).split("/", 1)
                if scene not in scene_list:
                    continue
                batch = self._build_visualization_batch(scene, frame_id)
                if batch is not None:
                    yield batch

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

            num_target_views = self.view_sampler.num_target_views
            if len(target_indices) < num_target_views:
                continue

            perm = torch.randperm(len(target_indices))[:num_target_views]
            sampled_target_indices = [target_indices[i] for i in perm.tolist()]

            batch = self._build_example(
                scene, context_indices, sampled_target_indices, overlap
            )
            if batch is not None:
                yield batch

    def get_bound(self, bound: str, num_views: int) -> torch.Tensor:
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


DATASET_CLASS = DatasetScannetDistill
DATASET_NAMES = ("scannet_distill",)
CFG_WRAPPERS = (DatasetScannetDistillCfgWrapper,)
