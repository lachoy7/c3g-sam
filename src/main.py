from pathlib import Path

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf

from src.misc.weight_modify import checkpoint_filter_fn
from src.model.encoder.common.gmae import remap_instill_to_qkv_checkpoint

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import (
        load_typed_root_config,
        val_check_interval_in_training_batches,
    )
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper
    from src.model.distillation_wrapper import (
        DistillationModelWrapper,
        DistillTrainCfg,
        DebugDecoderCfg,
        OptimizerCfg as DistillOptimizerCfg,
    )
    from src.model.load_foundation_model import load_foundation_model


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


def uses_precomputed_sam_features(cfg_dict: DictConfig) -> bool:
    for group in cfg_dict.get("dataset", []):
        if not isinstance(group, DictConfig):
            continue
        for val in group.values():
            if isinstance(val, DictConfig) and val.get("sam_features_root"):
                return True
    return False


def configure_prompted_sam_unfrozen_training(cfg_dict: DictConfig) -> None:
    """Train all C3G components; keep only SAM encoder and mask decoder frozen."""
    if cfg_dict.get("train", {}).get("prompt_mode") != "prompted":
        return

    OmegaConf.set_struct(cfg_dict, False)
    cfg_dict.model.encoder.freeze_backbone = False
    cfg_dict.model.encoder.freeze_instill_qk = False
    cfg_dict.model.encoder.freeze_geometry_head = False
    cfg_dict.model.decoder.feature_detach = False
    OmegaConf.set_struct(cfg_dict, True)
    print(
        cyan(
            "Prompted SAM: training full C3G encoder/decoder "
            "(SAM image encoder + mask decoder stay frozen)."
        )
    )


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    configure_prompted_sam_unfrozen_training(cfg_dict)
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(cyan(f"Saving outputs to {output_dir}."))

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            entity=cfg_dict.wandb.entity,
            mode=cfg_dict.wandb.mode,
            name=cfg_dict.wandb.name,
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
            print(cyan(f"W&B run: {wandb.run.url}"))
    else:
        logger = LocalLogger()

    # Set up checkpointing.
    checkpoint_kwargs: dict = {
        "dirpath": output_dir / "checkpoints",
        "save_top_k": cfg.checkpointing.save_top_k,
        "save_weights_only": cfg.checkpointing.save_weights_only,
        "monitor": cfg.checkpointing.monitor,
        "mode": cfg.checkpointing.mode,
        "save_last": True,
    }
    if cfg.checkpointing.every_n_train_steps is not None:
        checkpoint_kwargs["every_n_train_steps"] = cfg.checkpointing.every_n_train_steps
    checkpoint_callback = ModelCheckpoint(**checkpoint_kwargs)
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = "_"
    callbacks.append(checkpoint_callback)
    print(
        cyan(
            f"Checkpoints: monitor={cfg.checkpointing.monitor!r} "
            f"mode={cfg.checkpointing.mode!r}"
            + (
                f" every_n_train_steps={cfg.checkpointing.every_n_train_steps}"
                if cfg.checkpointing.every_n_train_steps is not None
                else " (validation-triggered saves only)"
            )
        )
    )

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    if cfg.trainer.devices is not None:
        trainer_devices = cfg.trainer.devices
    else:
        trainer_devices = "auto"

    use_ddp = (
        trainer_devices == "auto"
        and torch.cuda.device_count() > 1
    ) or (
        isinstance(trainer_devices, int)
        and trainer_devices > 1
    )

    trainer_kwargs: dict = dict(
        max_epochs=-1,
        num_nodes=cfg.trainer.num_nodes,
        accelerator="gpu",
        logger=logger,
        devices=trainer_devices,
        strategy=(
            DDPStrategy(
                find_unused_parameters=False,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
            if use_ddp
            else "auto"
        ),
        callbacks=callbacks,
        val_check_interval=val_check_interval_in_training_batches(
            cfg.trainer.val_check_interval,
            cfg.trainer.accumulate_grad_batches,
        ),
        check_val_every_n_epoch=None,
        enable_progress_bar=True,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        inference_mode=False if (cfg.mode == "test" and cfg.test.align_pose) else True,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )
    if cfg.trainer.limit_test_batches is not None:
        trainer_kwargs["limit_test_batches"] = cfg.trainer.limit_test_batches
    if cfg.trainer.num_sanity_val_steps is not None:
        trainer_kwargs["num_sanity_val_steps"] = cfg.trainer.num_sanity_val_steps
    trainer = Trainer(**trainer_kwargs)
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    use_distillation = cfg.train.pipeline == "distillation"

    if use_distillation:
        cfg.model.encoder.feature_dim = cfg.model.encoder.gaussian_feature_dim

        encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

        if cfg.model.encoder.pretrained_weights and cfg.mode == "train":
            weight_path = cfg.model.encoder.pretrained_weights
            ckpt_weights = torch.load(weight_path, map_location="cpu")
            if "model" in ckpt_weights:
                ckpt_weights = ckpt_weights["model"]
                ckpt_weights = checkpoint_filter_fn(ckpt_weights, encoder)
                ckpt_weights = remap_instill_to_qkv_checkpoint(ckpt_weights)
                missing_keys, unexpected_keys = encoder.load_state_dict(
                    ckpt_weights, strict=False
                )
            elif "state_dict" in ckpt_weights:
                ckpt_weights = ckpt_weights["state_dict"]
                ckpt_weights = {
                    k[8:]: v
                    for k, v in ckpt_weights.items()
                    if k.startswith("encoder.")
                }
                ckpt_weights = remap_instill_to_qkv_checkpoint(ckpt_weights)
                missing_keys, unexpected_keys = encoder.load_state_dict(
                    ckpt_weights, strict=False
                )
            elif isinstance(ckpt_weights, dict):
                new_ckpt = {}
                for key, value in ckpt_weights.items():
                    if "aggregator" in key:
                        new_ckpt[f"backbone.{key}"] = value
                    if "point_head" in key:
                        new_ckpt[key.replace("point_head", "dpt_head")] = value
                new_ckpt = remap_instill_to_qkv_checkpoint(new_ckpt)
                missing_keys, unexpected_keys = encoder.load_state_dict(
                    new_ckpt, strict=False
                )
                del new_ckpt
            else:
                raise ValueError(f"Invalid checkpoint format: {weight_path}")

            geometry_param_names = {
                "gaussian_token",
                "gaussian_tokens",
                "gmae_to_gaussians",
                "anchor_positions",
            }
            geometry_missing = [
                k for k in missing_keys if k.split(".", 1)[0] in geometry_param_names
            ]
            named_params = dict(encoder.named_parameters())
            frozen_geometry_missing = [
                k
                for k in geometry_missing
                if k in named_params and not named_params[k].requires_grad
            ]
            print(
                f"[load] missing={len(missing_keys)} unexpected={len(unexpected_keys)}"
            )
            print(
                f"  missing geometry keys ({len(geometry_missing)}):",
                geometry_missing[:10],
            )
            if frozen_geometry_missing:
                print(
                    f"  missing frozen geometry keys ({len(frozen_geometry_missing)}):",
                    frozen_geometry_missing[:10],
                )
            if unexpected_keys:
                print(f"  unexpected keys (first 10):", unexpected_keys[:10])
            if frozen_geometry_missing:
                raise RuntimeError(
                    f"FATAL: {len(frozen_geometry_missing)} frozen geometry keys "
                    f"missing from checkpoint '{weight_path}'! Frozen random-init "
                    f"params will produce garbage. Missing: "
                    f"{frozen_geometry_missing[:5]}"
                )

            del ckpt_weights

        distill_train_cfg = DistillTrainCfg(
            feature_cosine_loss_weight=cfg_dict.train.get(
                "feature_cosine_loss_weight", 1.0
            ),
            feature_mag_loss_weight=cfg_dict.train.get("feature_mag_loss_weight", 0.1),
            depth_mode=cfg.train.depth_mode,
            context_view_loss=cfg.train.context_view_loss,
        )
        distill_optimizer_cfg = DistillOptimizerCfg(
            lr=cfg.optimizer.lr,
            warm_up_steps=cfg.optimizer.warm_up_steps,
            weight_decay=cfg_dict.optimizer.get("weight_decay", 0.05),
            feature_head_weight_decay=cfg_dict.optimizer.get(
                "feature_head_weight_decay", 0.01
            ),
        )

        debug_decoder_cfg = None
        if cfg_dict.get("debug_decoder", {}).get("enabled", False):
            debug_decoder_cfg = DebugDecoderCfg(
                enabled=True,
                sam_checkpoint=cfg_dict.debug_decoder.get(
                    "sam_checkpoint", "./pretrained_weights/sam_vit_h.pth"
                ),
                sam_model_variant=cfg_dict.debug_decoder.get(
                    "sam_model_variant", "sam_vit_h"
                ),
            )

        model_wrapper = DistillationModelWrapper(
            distill_optimizer_cfg,
            distill_train_cfg,
            encoder,
            get_decoder(cfg.model.decoder),
            get_losses(cfg.loss),
            step_tracker,
            debug_decoder_cfg=debug_decoder_cfg,
        )
    else:
        skip_sam_encoder = uses_precomputed_sam_features(
            cfg_dict
        ) and "sam" in cfg.train.reproj_model
        if skip_sam_encoder:
            print(cyan("Using precomputed SAM features; skipping SAM encoder load."))
        vggt, dino, lseg_feature_extractor, clip, sam_encoder, feature_dim = (
            load_foundation_model(cfg, skip_sam_encoder=skip_sam_encoder)
        )
        cfg.model.encoder.feature_dim = (
            feature_dim if cfg.train.feature_rendering_loss > 0 else 0
        )

        encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

        if cfg.model.encoder.pretrained_weights and cfg.mode == "train":
            weight_path = cfg.model.encoder.pretrained_weights
            ckpt_weights = torch.load(weight_path, map_location="cpu")
            if "model" in ckpt_weights:
                ckpt_weights = ckpt_weights["model"]
                ckpt_weights = checkpoint_filter_fn(ckpt_weights, encoder)
                ckpt_weights = remap_instill_to_qkv_checkpoint(ckpt_weights)
                missing_keys, unexpected_keys = encoder.load_state_dict(
                    ckpt_weights, strict=False
                )
            elif "state_dict" in ckpt_weights:
                ckpt_weights = ckpt_weights["state_dict"]
                ckpt_weights = {
                    k[8:]: v
                    for k, v in ckpt_weights.items()
                    if k.startswith("encoder.")
                }
                ckpt_weights = remap_instill_to_qkv_checkpoint(ckpt_weights)
                missing_keys, unexpected_keys = encoder.load_state_dict(
                    ckpt_weights, strict=False
                )
            elif isinstance(ckpt_weights, dict):
                new_ckpt = {}
                for key, value in ckpt_weights.items():
                    if "aggregator" in key:
                        new_ckpt[f"backbone.{key}"] = value
                    if "point_head" in key:
                        new_ckpt[key.replace("point_head", "dpt_head")] = value
                new_ckpt = remap_instill_to_qkv_checkpoint(new_ckpt)
                missing_keys, unexpected_keys = encoder.load_state_dict(
                    new_ckpt, strict=False
                )
                del new_ckpt
            else:
                raise ValueError(f"Invalid checkpoint format: {weight_path}")

            print(
                f"[load] missing={len(missing_keys)} unexpected={len(unexpected_keys)}"
            )
            if missing_keys:
                print(f"  missing keys (first 10):", missing_keys[:10])
            if unexpected_keys:
                print(f"  unexpected keys (first 10):", unexpected_keys[:10])

            del ckpt_weights

        debug_decoder_cfg = None
        if cfg_dict.get("debug_decoder", {}).get("enabled", False):
            debug_decoder_cfg = DebugDecoderCfg(
                enabled=True,
                sam_checkpoint=cfg_dict.debug_decoder.get(
                    "sam_checkpoint", cfg.train.sam_checkpoint
                ),
                sam_model_variant=cfg_dict.debug_decoder.get(
                    "sam_model_variant", cfg.train.sam_model_variant
                ),
            )

        model_wrapper = ModelWrapper(
            cfg.optimizer,
            cfg.test,
            cfg.train,
            encoder,
            encoder_visualizer,
            get_decoder(cfg.model.decoder),
            get_losses(cfg.loss),
            step_tracker,
            vggt=vggt,
            dino=dino,
            clip=clip,
            lseg_feature_extractor=lseg_feature_extractor,
            sam_encoder=sam_encoder,
            mode=cfg.mode,
            debug_decoder_cfg=debug_decoder_cfg,
        )
        if cfg.train.prompt_mode == "prompted":
            if sam_encoder is not None:
                for param in sam_encoder.parameters():
                    param.requires_grad = False
                sam_encoder.eval()
            prompted_loss = getattr(
                model_wrapper, "prompted_segmentation_loss", None
            )
            if prompted_loss is not None:
                for param in prompted_loss.mask_decoder.parameters():
                    param.requires_grad = False
                prompted_loss.mask_decoder.eval()
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )
    torch.cuda.empty_cache()

    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    import sys

    if "--modal" in sys.argv:
        modal_argv = sys.argv[sys.argv.index("--modal") + 1 :]
        from src.modal.launch import run_modal_cli

        run_modal_cli(modal_argv)
    else:
        train()
