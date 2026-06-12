"""ScanNet 2D semantic segmentation loader (flat per-scene layout).

Expects data prepared by :mod:`download_scannet` on the Modal ``scannet`` volume::

    <root>/<scene_id>/{frame_id}_x.jpg
    <root>/<scene_id>/{frame_id}_cam.npz
    <root>/<scene_id>/{frame_id}_y.png

Sampling, preprocessing, and batch layout match :mod:`dataset_replica_2dseg`
and :mod:`dataset_replica_semseg`; only scene ids and on-disk paths differ.

For C3G-SAM dual-resolution, each view also includes ``sam_image``: source RGB
resized directly to the SAM encoder size (1024×1024), independent of
``input_image_shape`` (252×252 for VGGT/splatting). When ``sam_features_root`` is
set, precomputed ``{frame_id}_sam.pt`` tensors (256×64×64) are loaded instead and
``sam_image`` is omitted.

Each yielded sample has ``view_sampler.num_context_views`` context frames (typically
2) and ``view_sampler.num_target_views`` target frames. When the sampler returns
more targets than that (e.g. test mode), the dataset subsamples without replacement;
scenes with fewer candidates are skipped. Configure targets via
``dataset.scannet_2dseg.view_sampler.num_target_views`` only.
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
from ..model.sam.constants import SAM_IMAGE_SIZE
from .dataset import DatasetCfgCommon
from .scannet_2dseg_splits import discover_scene_ids, scenes_for_stage
from .view_sampler import ViewSampler

logger = logging.getLogger(__name__)


@dataclass
class Scannet2dSegCfg(DatasetCfgCommon):
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
    prompt_strategy: str = "centroid"
    min_object_pixels: int = 16
    sam_features_root: Path | None = None


@dataclass
class DatasetScannet2dSegCfgWrapper:
    scannet_2dseg: Scannet2dSegCfg


class DatasetScannet2dSeg(IterableDataset):
    """Loads ScanNet flat 2D-seg volume with the same logic as ``DatasetReplica2dSeg``."""

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
        self.intrinsics = self.load_intrinsics()
        self.frame_ids = {
            scene: list_frame_ids(self.root / scene) for scene in self.scenes
        }

    def load_intrinsics(self):
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

    @property
    def use_precomputed_sam_features(self) -> bool:
        return self.cfg.sam_features_root is not None

    def _sam_feature_path(self, scene: str, frame_id: str) -> Path:
        assert self.cfg.sam_features_root is not None
        return Path(self.cfg.sam_features_root) / scene / f"{frame_id}_sam.pt"

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
        scene_list = list(self.scenes)

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

            num_target_views = self.view_sampler.num_target_views
            if len(target_indices) < num_target_views:
                continue

            perm = torch.randperm(len(target_indices))[:num_target_views]
            sampled_target_indices = [target_indices[i] for i in perm.tolist()]
            idxs = list(context_indices) + sampled_target_indices

            extrinsics_list = []
            intrinsics_list = []
            images_list = []
            sam_images_list: list[torch.Tensor] = []
            sam_features_list: list[torch.Tensor] = []
            label_list = []
            valid = True
            use_precomputed = self.use_precomputed_sam_features

            for view_index in idxs:
                frame_id = frame_ids[view_index]
                paths = FramePaths.from_frame_id(self.root / scene, frame_id)

                if not paths.image.is_file() or not paths.label.is_file():
                    logger.warning(f"Missing files for {scene} frame {frame_id}")
                    valid = False
                    break

                if use_precomputed:
                    sam_path = self._sam_feature_path(scene, frame_id)
                    if not sam_path.is_file():
                        logger.warning(
                            f"Missing SAM features for {scene} frame {frame_id}: "
                            f"{sam_path}"
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
                if not use_precomputed:
                    rgb_sam = cv2.resize(
                        rgb,
                        (SAM_IMAGE_SIZE, SAM_IMAGE_SIZE),
                        interpolation=cv2.INTER_LINEAR,
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
                if use_precomputed:
                    sam_feat = torch.load(
                        self._sam_feature_path(scene, frame_id), map_location="cpu"
                    )
                    if sam_feat.shape != (256, 64, 64):
                        logger.warning(
                            f"Unexpected SAM feature shape {sam_feat.shape} for "
                            f"{scene} frame {frame_id}, expected (256, 64, 64)"
                        )
                        valid = False
                        break
                    sam_features_list.append(sam_feat)
                else:
                    sam_images_list.append(self.to_tensor(Image.fromarray(rgb_sam)))
                label_list.append(
                    torch.from_numpy(label_resized.astype(np.int64))
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
            labels = torch.stack(label_list, dim=0)
            if use_precomputed:
                sam_features = torch.stack(sam_features_list, dim=0)
            else:
                sam_images = torch.stack(sam_images_list, dim=0)

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
            target_frame_ids = [int(frame_ids[i]) for i in sampled_target_indices]

            context_views: dict = {
                "extrinsics": extrinsics[:num_ctx],
                "intrinsics": intrinsics[:num_ctx],
                "image": images[:num_ctx],
                "label": labels[:num_ctx],
                "near": self.get_bound("near", num_ctx) / scale,
                "far": self.get_bound("far", num_ctx) / scale,
                "index": torch.tensor(context_frame_ids, dtype=torch.int64),
                "overlap": overlap,
            }
            target_views: dict = {
                "extrinsics": extrinsics[num_ctx:],
                "intrinsics": intrinsics[num_ctx:],
                "image": images[num_ctx:],
                "label": labels[num_ctx:],
                "near": self.get_bound("near", num_target_views) / scale,
                "far": self.get_bound("far", num_target_views) / scale,
                "index": torch.tensor(target_frame_ids, dtype=torch.int64),
            }
            if use_precomputed:
                context_views["sam_features"] = sam_features[:num_ctx]
                target_views["sam_features"] = sam_features[num_ctx:]
            else:
                context_views["sam_image"] = sam_images[:num_ctx]
                target_views["sam_image"] = sam_images[num_ctx:]

            yield {
                "context": context_views,
                "target": target_views,
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


DATASET_CLASS = DatasetScannet2dSeg
DATASET_NAMES = ("scannet_2dseg",)
CFG_WRAPPERS = (DatasetScannet2dSegCfgWrapper,)
