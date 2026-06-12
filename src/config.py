from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

from .dataset import DatasetCfgWrapper
from .dataset.data_module import DataLoaderCfg
from .loss import LossCfgWrapper
from .model.decoder import DecoderCfg
from .model.encoder import EncoderCfg
from .model.model_wrapper import OptimizerCfg, TestCfg, TrainCfg


@dataclass
class CheckpointingCfg:
    load: Optional[str]  # Not a path, since it could be something like wandb://...
    save_top_k: int
    save_weights_only: bool
    # When set, also save on this train-step interval. Omit/null for val-metric-only saves.
    every_n_train_steps: Optional[int] = None
    monitor: str = "info/global_step"
    mode: Literal["min", "max"] = "max"
    debug_log_interval: int = 50


@dataclass
class ModelCfg:
    decoder: DecoderCfg
    encoder: EncoderCfg


@dataclass
class TrainerCfg:
    max_steps: int
    # Optimizer steps between validation runs (int), or fraction of a train epoch (float).
    val_check_interval: int | float | None
    gradient_clip_val: int | float | None
    num_nodes: int = 1
    accumulate_grad_batches: int = 1
    limit_test_batches: int | float | None = None
    # How often Lightning flushes self.log metrics to W&B (optimizer steps).
    log_every_n_steps: int = 10
    # Lightning ``devices`` (e.g. 1 or "auto"). None keeps legacy "auto" + DDP behavior.
    devices: int | str | None = None
    # Set to 0 to skip sanity validation (useful on multi-GPU hosts).
    num_sanity_val_steps: int | None = None


def val_check_interval_in_training_batches(
    val_check_interval: int | float | None,
    accumulate_grad_batches: int,
) -> int | float | None:
    """Map config val_check_interval (optimizer steps) to Lightning training batches."""
    if val_check_interval is None or not isinstance(val_check_interval, int):
        return val_check_interval
    return val_check_interval * accumulate_grad_batches


@dataclass
class RootCfg:
    wandb: dict
    mode: Literal["train", "test"]
    dataset: list[DatasetCfgWrapper]
    data_loader: DataLoaderCfg
    model: ModelCfg
    optimizer: OptimizerCfg
    checkpointing: CheckpointingCfg
    trainer: TrainerCfg
    loss: list[LossCfgWrapper]
    test: TestCfg
    train: TrainCfg
    seed: int


TYPE_HOOKS = {
    Path: Path,
}


T = TypeVar("T")


def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict = {},
) -> T:
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg),
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}),
    )


def separate_loss_cfg_wrappers(joined: dict) -> list[LossCfgWrapper]:
    # The dummy allows the union to be converted.
    @dataclass
    class Dummy:
        dummy: LossCfgWrapper

    return [
        load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy
        for k, v in joined.items()
    ]


def separate_dataset_cfg_wrappers(joined: dict) -> list[DatasetCfgWrapper]:
    # The dummy allows the union to be converted.
    @dataclass
    class Dummy:
        dummy: DatasetCfgWrapper

    return [
        load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy
        for k, v in joined.items()
    ]


def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    return load_typed_config(
        cfg,
        RootCfg,
        {
            list[LossCfgWrapper]: separate_loss_cfg_wrappers,
            list[DatasetCfgWrapper]: separate_dataset_cfg_wrappers,
        },
    )
