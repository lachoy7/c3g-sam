from dataclasses import fields

from torch.utils.data import Dataset

from .dataset_scannet_pose import DatasetScannetPose, DatasetScannetPoseCfgWrapper

from .dataset_replica_2dseg import (
    DatasetReplica2dSeg,
    Replica2dSegCfg,
    DatasetReplica2dSegCfgWrapper,
)
from .dataset_replica_distill import (
    DatasetReplicaDistill,
    ReplicaDistillCfg,
    DatasetReplicaDistillCfgWrapper,
)
from .dataset_replica_semseg import (
    DatasetReplicaSemSeg,
    ReplicaSemSegCfg,
    DatasetReplicaSemSegCfgWrapper,
)
from ..misc.step_tracker import StepTracker
from .dataset_re10k import (
    DatasetRE10k,
    DatasetRE10kCfg,
    DatasetRE10kCfgWrapper,
    DatasetDL3DVCfgWrapper,
    DatasetScannetppCfgWrapper,
)
from .dataset_scannet import ScannetCfg, DatasetScannet, DatasetScannetCfgWrapper
from .dataset_scannet_2dseg import (
    DatasetScannet2dSeg,
    Scannet2dSegCfg,
    DatasetScannet2dSegCfgWrapper,
)
from .dataset_scannet_distill import (
    DatasetScannetDistill,
    ScannetDistillCfg,
    DatasetScannetDistillCfgWrapper,
)
from .dataset_replica import DatasetReplica, ReplicaCfg, DatasetReplicaCfgWrapper
from .dataset_lerf_mask import (
    DatasetLerfMask,
    LerfMaskCfg,
    DatasetLerfMaskCfgWrapper,
)
from .types import Stage
from .view_sampler import get_view_sampler

DATASETS: dict[str, Dataset] = {
    "re10k": DatasetRE10k,
    "dl3dv": DatasetRE10k,
    "scannetpp": DatasetRE10k,
    "scannet_pose": DatasetScannetPose,
    "scannet": DatasetScannet,
    "scannet_2dseg": DatasetScannet2dSeg,
    "scannet_distill": DatasetScannetDistill,
    "replica": DatasetReplica,
    "replica_2dseg": DatasetReplica2dSeg,
    "replica_distill": DatasetReplicaDistill,
    "replica_semseg": DatasetReplicaSemSeg,
    "lerf_mask": DatasetLerfMask,
}


DatasetCfgWrapper = (
    DatasetRE10kCfgWrapper
    | DatasetDL3DVCfgWrapper
    | DatasetScannetppCfgWrapper
    | DatasetScannetPoseCfgWrapper
    | DatasetScannetCfgWrapper
    | DatasetScannet2dSegCfgWrapper
    | DatasetScannetDistillCfgWrapper
    | DatasetReplicaCfgWrapper
    | DatasetReplica2dSegCfgWrapper
    | DatasetReplicaDistillCfgWrapper
    | DatasetReplicaSemSegCfgWrapper
    | DatasetLerfMaskCfgWrapper
)
DatasetCfg = (
    DatasetRE10kCfg
    | ScannetCfg
    | Scannet2dSegCfg
    | ScannetDistillCfg
    | ReplicaCfg
    | Replica2dSegCfg
    | ReplicaDistillCfg
    | ReplicaSemSegCfg
    | LerfMaskCfg
)


def get_dataset(
    cfgs: list[DatasetCfgWrapper],
    stage: Stage,
    step_tracker: StepTracker | None,
) -> list[Dataset]:
    datasets = []
    for cfg in cfgs:
        (field,) = fields(type(cfg))
        cfg = getattr(cfg, field.name)

        if cfg.name == "lerf_mask" and stage != "test":
            raise ValueError(
                "LERF-Mask is evaluation-only. Use mode=test with +evaluation=lerf_mask."
            )

        view_sampler = get_view_sampler(
            cfg.view_sampler,
            stage,
            cfg.overfit_to_scene is not None,
            cfg.cameras_are_circular,
            step_tracker,
        )
        dataset = DATASETS[cfg.name](cfg, stage, view_sampler)
        datasets.append(dataset)

    return datasets
