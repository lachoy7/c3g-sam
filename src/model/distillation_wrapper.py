from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import rank_zero_only
from torch import nn

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.frame_layout import format_frame_id
from ..misc.step_tracker import StepTracker
from .decoder.decoder import Decoder, DepthRenderingMode
from .debug_visualizer import log_debug_visualizations, log_decoder_debug
from .encoder.encoder_vggt import EncoderVGGT
from .types import Gaussians


def compute_feature_losses(rendered, target, eps: float = 1e-8):
    """Cosine similarity loss between rendered and target features."""
    rendered_norm = rendered.norm(dim=2, keepdim=True)
    target_norm = target.norm(dim=2, keepdim=True)

    cosine_sim = (
        (rendered / rendered_norm.clamp(min=eps))
        * (target / target_norm.clamp(min=eps))
    ).sum(dim=2)
    cosine_loss = (1.0 - cosine_sim).mean()

    return cosine_loss


@dataclass
class DistillTrainCfg:
    feature_cosine_loss_weight: float = 1.0
    feature_mag_loss_weight: float = 0.1
    depth_mode: DepthRenderingMode | None = None
    context_view_loss: bool = True
    random_select_context_view: bool = False


@dataclass
class DebugDecoderCfg:
    enabled: bool = False
    sam_checkpoint: str = "./pretrained_weights/sam_vit_h.pth"
    sam_model_variant: str = "sam_vit_h"


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    weight_decay: float = 0.05
    feature_head_weight_decay: float = 0.01


class DistillationModelWrapper(LightningModule):
    """Training loop for MSE feature distillation using pre-computed SAM features."""

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        train_cfg: DistillTrainCfg,
        encoder: EncoderVGGT,
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        debug_decoder_cfg: DebugDecoderCfg | None = None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker
        self.encoder = encoder
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)
        self._last_train_debug_step_logged: int | None = None
        self._logged_test_visualization_keys: set[str] = set()

        self.sam_debug_decoder = None
        if debug_decoder_cfg and debug_decoder_cfg.enabled:
            from .sam_decoder import SAMMaskDecoderWrapper

            self.sam_debug_decoder = SAMMaskDecoderWrapper(
                debug_decoder_cfg.sam_checkpoint,
                model_variant=debug_decoder_cfg.sam_model_variant,
            )
            self.sam_debug_decoder.eval()
            for p in self.sam_debug_decoder.parameters():
                p.requires_grad = False

    @rank_zero_only
    def on_train_start(self) -> None:
        accum = self.trainer.accumulate_grad_batches or 1
        print(
            "Distillation training started: "
            f"max_steps={self.trainer.max_steps}, "
            f"accumulate_grad_batches={accum}, "
            f"log_every_n_steps={self.trainer.log_every_n_steps}",
            flush=True,
        )

    def _downsample_for_encoder(self, sam_features, h, w):
        """Downsample SAM features from 64x64 to patch resolution for the encoder.

        The InstillTransformer requires the context_feature sequence length to
        match the backbone patch-token sequence length.  The backbone produces
        (H/patch_size)*(W/patch_size) tokens per view, while raw SAM features
        have 64*64 = 4096 tokens per view.  We bilinearly interpolate to bridge
        this gap (same as the live-SAM pipeline does in
        ModelWrapper.forward_foundation_model with ``interpolate=True``).
        """
        b, v, c, fh, fw = sam_features.shape
        patch_size = self.encoder.patch_size
        patch_h = h // patch_size
        patch_w = w // patch_size
        if patch_h <= 0 or patch_w <= 0:
            raise ValueError(
                f"Cannot downsample SAM features: image ({h}, {w}) is smaller than "
                f"patch_size={patch_size}"
            )
        if fh == patch_h and fw == patch_w:
            return sam_features
        flat = rearrange(sam_features, "b v c h w -> (b v) c h w")
        flat = F.interpolate(
            flat, size=(patch_h, patch_w), mode="bilinear", align_corners=False
        )
        return rearrange(flat, "(b v) c h w -> b v c h w", b=b, v=v)

    def _validate_distill_batch_shapes(
        self, batch: BatchedExample, h: int, w: int
    ) -> None:
        """Catch layout bugs early (bad collate / wrong tensor dims → CUDA OOM)."""
        if h <= 0 or w <= 0 or h > 4096 or w > 4096:
            raise ValueError(f"Suspicious target image size ({h}, {w})")
        for stage in ("context", "target"):
            img = batch[stage]["image"]
            if img.shape[-2:] != (h, w):
                raise ValueError(
                    f"{stage} image spatial {tuple(img.shape[-2:])} != target ({h}, {w}); "
                    f"full shape {tuple(img.shape)}"
                )
            sam = batch[stage]["sam_features"]
            if sam.ndim != 5:
                raise ValueError(
                    f"{stage} sam_features must be 5D (B,V,C,H,W), got {tuple(sam.shape)}"
                )

    def _should_log_debug_visualizations(self) -> bool:
        trainer = getattr(self, "trainer", None)
        if trainer is not None and getattr(trainer, "sanity_checking", False):
            return False
        return True

    def _get_valid_region(self, sam_features):
        """Return valid (non-padding) region size in the 64x64 SAM embedding space."""
        return sam_features.shape[-2], sam_features.shape[-1]

    @staticmethod
    def _batch_visualization_key(batch: BatchedExample) -> str:
        scene = batch["scene"]
        if isinstance(scene, (list, tuple)):
            scene = scene[0]
        target_index = int(batch["target"]["index"][0, 0].item())
        return f"{scene}/{format_frame_id(target_index)}"

    def _configured_visualization_keys(self) -> set[str]:
        eval_cfg = get_cfg().get("eval", {})
        keys = eval_cfg.get("visualization_keys")
        if keys is None:
            return set()
        return {str(key) for key in keys}

    def _matching_visualization_key(self, batch: BatchedExample) -> str | None:
        viz_keys = self._configured_visualization_keys()
        if not viz_keys:
            return None
        scene = batch["scene"]
        if isinstance(scene, (list, tuple)):
            scene = scene[0]
        for target_index in batch["target"]["index"][0].tolist():
            key = f"{scene}/{format_frame_id(int(target_index))}"
            if key in viz_keys:
                return key
        return None

    def _should_log_test_visualization(self, batch: BatchedExample) -> bool:
        if not self._should_log_debug_visualizations():
            return False
        key = self._matching_visualization_key(batch)
        if key is None or key in self._logged_test_visualization_keys:
            return False
        self._logged_test_visualization_keys.add(key)
        return True

    def _distill_forward_eval(
        self, batch: BatchedExample, h: int, w: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, object, int, int]:
        context_sam = batch["context"]["sam_features"]
        target_sam = batch["target"]["sam_features"]
        context_sam_enc = self._downsample_for_encoder(context_sam, h, w)

        gaussians = self.encoder(
            batch["context"],
            self.global_step,
            context_feature=context_sam_enc,
        )

        gaussians_detached = Gaussians(
            means=gaussians.means.detach(),
            covariances=gaussians.covariances.detach(),
            harmonics=gaussians.harmonics.detach(),
            opacities=gaussians.opacities.detach(),
            feature=gaussians.feature,
        )

        output = self.decoder.forward(
            gaussians_detached,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )

        b, v, c, fh, fw = target_sam.shape
        rendered_interp = F.interpolate(
            rearrange(output.feature, "b v c h w -> (b v) c h w"),
            size=(fh, fw),
            mode="bilinear",
            align_corners=False,
        )
        rendered_interp = rearrange(
            rendered_interp, "(b v) c h w -> b v c h w", b=b, v=v
        )

        valid_h, valid_w = self._get_valid_region(target_sam)
        rendered_crop = rendered_interp[:, :, :, :valid_h, :valid_w]
        target_crop = target_sam[:, :, :, :valid_h, :valid_w]
        return (
            target_sam,
            rendered_interp,
            rendered_crop,
            target_crop,
            output,
            valid_h,
            valid_w,
        )

    def _log_distill_debug_visualizations(
        self,
        batch: BatchedExample,
        *,
        target_sam: torch.Tensor,
        rendered_interp: torch.Tensor,
        output,
        h: int,
        w: int,
        valid_h: int,
        valid_w: int,
        prefix: str,
        log_interval: int = 1,
    ) -> None:
        if self.global_rank != 0:
            return
        log_debug_visualizations(
            self.logger,
            self.global_step,
            log_interval,
            batch["target"]["image"][0, 0],
            target_sam[0, 0],
            rendered_interp[0, 0],
            (h, w),
            prefix=prefix,
            valid_region=(valid_h, valid_w),
            rendered_rgb=output.color[0, 0].detach().float()
            if output.color is not None
            else None,
        )
        log_decoder_debug(
            self.logger,
            self.global_step,
            log_interval,
            self.sam_debug_decoder,
            target_sam[0, 0].detach().float(),
            rendered_interp[0, 0].detach().float(),
            batch["target"]["image"][0, 0].detach().float(),
            (h, w),
            prefix=prefix,
            valid_region=(valid_h, valid_w),
        )

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape
        self._validate_distill_batch_shapes(batch, h, w)

        context_sam = batch["context"]["sam_features"]
        target_sam = batch["target"]["sam_features"]

        # Downsample context features to patch resolution for the encoder.
        # Loss targets remain at full 64x64 SAM resolution.
        context_sam_enc = self._downsample_for_encoder(context_sam, h, w)

        gaussians = self.encoder(
            batch["context"],
            self.global_step,
            context_feature=context_sam_enc,
        )

        gaussians_detached = Gaussians(
            means=gaussians.means.detach(),
            covariances=gaussians.covariances.detach(),
            harmonics=gaussians.harmonics.detach(),
            opacities=gaussians.opacities.detach(),
            feature=gaussians.feature,
        )

        if self.train_cfg.context_view_loss:
            extrinsics = torch.cat(
                [batch["target"]["extrinsics"], batch["context"]["extrinsics"]], dim=1
            )
            intrinsics = torch.cat(
                [batch["target"]["intrinsics"], batch["context"]["intrinsics"]], dim=1
            )
            near = torch.cat([batch["target"]["near"], batch["context"]["near"]], dim=1)
            far = torch.cat([batch["target"]["far"], batch["context"]["far"]], dim=1)
        else:
            extrinsics = batch["target"]["extrinsics"]
            intrinsics = batch["target"]["intrinsics"]
            near = batch["target"]["near"]
            far = batch["target"]["far"]

        output = self.decoder.forward(
            gaussians_detached,
            extrinsics,
            intrinsics,
            near,
            far,
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )

        if self.train_cfg.context_view_loss:
            all_sam = torch.cat([target_sam, context_sam], dim=1)
        else:
            all_sam = target_sam

        b, v_total, c_feat, fh, fw = all_sam.shape
        rendered_features = output.feature
        rendered_interp = F.interpolate(
            rearrange(rendered_features, "b v c h w -> (b v) c h w"),
            size=(fh, fw),
            mode="bilinear",
            align_corners=False,
        )
        rendered_interp = rearrange(
            rendered_interp, "(b v) c h w -> b v c h w", b=b, v=v_total
        )

        valid_h, valid_w = self._get_valid_region(all_sam)
        rendered_crop = rendered_interp[:, :, :, :valid_h, :valid_w]
        target_crop = all_sam[:, :, :, :valid_h, :valid_w]

        with torch.no_grad():
            current_norm = target_crop.norm(dim=2).mean()
            m = self.encoder.feature_norm_ema_momentum
            new_norm = m * self.encoder.feature_norm_ema + (1 - m) * current_norm
            self.encoder.feature_norm_ema.copy_(new_norm)

        feature_cosine_loss = compute_feature_losses(rendered_crop, target_crop)

        rendered_mag = rendered_crop.norm(dim=2)
        target_mag = target_crop.norm(dim=2)
        feature_mag_loss = F.l1_loss(rendered_mag, target_mag)

        total_loss = (
            self.train_cfg.feature_cosine_loss_weight * feature_cosine_loss
            + self.train_cfg.feature_mag_loss_weight * feature_mag_loss
        )

        for loss_name, loss_value in (
            ("loss/feature_cosine", feature_cosine_loss),
            ("loss/feature_magnitude", feature_mag_loss),
        ):
            self.log(
                loss_name,
                loss_value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        if self.train_cfg.context_view_loss:
            target_gt = torch.cat(
                [batch["target"]["image"], ((batch["context"]["image"] + 1) / 2)], dim=1
            )
        else:
            target_gt = batch["target"]["image"]

        for loss_fn in self.losses:
            loss = loss_fn.forward(
                output,
                batch,
                gaussians_detached,
                self.global_step,
                target_image=target_gt,
            )
            self.log(
                f"loss/{loss_fn.name}",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            total_loss = total_loss + loss

        self.log(
            "loss/total",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        ckpt_cfg = get_cfg()["checkpointing"]
        checkpoint_interval = ckpt_cfg.get(
            "debug_log_interval", ckpt_cfg.get("every_n_train_steps", 50)
        )
        should_log_train_debug = (
            self.global_rank == 0
            and self.global_step % checkpoint_interval == 0
            and self._last_train_debug_step_logged != self.global_step
        )
        if should_log_train_debug and self._should_log_debug_visualizations():
            self._last_train_debug_step_logged = self.global_step
            log_debug_visualizations(
                self.logger,
                self.global_step,
                checkpoint_interval,
                batch["target"]["image"][0, 0].detach().float(),
                target_sam[0, 0].detach().float(),
                rendered_interp[0, 0].detach().float(),
                (h, w),
                prefix="train",
                valid_region=(valid_h, valid_w),
                rendered_rgb=output.color[0, 0].detach().float()
                if output.color is not None
                else None,
            )
            log_decoder_debug(
                self.logger,
                self.global_step,
                checkpoint_interval,
                self.sam_debug_decoder,
                target_sam[0, 0].detach().float(),
                rendered_interp[0, 0].detach().float(),
                batch["target"]["image"][0, 0].detach().float(),
                (h, w),
                prefix="train",
                valid_region=(valid_h, valid_w),
            )
            print(
                f"train step {self.global_step} finished; "
                f"loss = {total_loss.detach().item():.6f}; "
                f"feature_cosine = {feature_cosine_loss.detach().item():.6f}; "
                f"feature_mag = {feature_mag_loss.detach().item():.6f}",
                flush=True,
            )

        self.log(
            "info/global_step",
            self.global_step,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return total_loss

    def _log_distill_metrics(
        self,
        rendered_crop: torch.Tensor,
        target_crop: torch.Tensor,
        *,
        prefix: Literal["val", "test"],
    ) -> None:
        feature_mag = F.l1_loss(rendered_crop.norm(dim=2), target_crop.norm(dim=2))
        feature_cosine = compute_feature_losses(rendered_crop, target_crop)
        self.log(
            f"{prefix}/feature_mag",
            feature_mag,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{prefix}/feature_cosine",
            feature_cosine,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape
        self._validate_distill_batch_shapes(batch, h, w)

        (
            target_sam,
            rendered_interp,
            rendered_crop,
            target_crop,
            output,
            valid_h,
            valid_w,
        ) = self._distill_forward_eval(batch, h, w)
        self._log_distill_metrics(rendered_crop, target_crop, prefix="val")

        if self.global_rank == 0 and self._should_log_debug_visualizations():
            ckpt_cfg = get_cfg()["checkpointing"]
            debug_interval = ckpt_cfg.get(
                "debug_log_interval", ckpt_cfg.get("every_n_train_steps", 50)
            )
            self._log_distill_debug_visualizations(
                batch,
                target_sam=target_sam,
                rendered_interp=rendered_interp,
                output=output,
                h=h,
                w=w,
                valid_h=valid_h,
                valid_w=valid_w,
                prefix="val",
                log_interval=debug_interval,
            )

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape
        self._validate_distill_batch_shapes(batch, h, w)

        (
            target_sam,
            rendered_interp,
            rendered_crop,
            target_crop,
            output,
            valid_h,
            valid_w,
        ) = self._distill_forward_eval(batch, h, w)
        self._log_distill_metrics(rendered_crop, target_crop, prefix="test")

        if batch_idx % 50 == 0 and self.global_rank == 0:
            print(
                f"test batch {batch_idx}; scene = {batch['scene']}; "
                f"target = {batch['target']['index'].tolist()}",
                flush=True,
            )

        if self._should_log_test_visualization(batch):
            viz_key = self._matching_visualization_key(batch)
            print(f"Logging test debug visualizations for {viz_key}", flush=True)
            self._log_distill_debug_visualizations(
                batch,
                target_sam=target_sam,
                rendered_interp=rendered_interp,
                output=output,
                h=h,
                w=w,
                valid_h=valid_h,
                valid_w=valid_w,
                prefix="test",
                log_interval=1,
            )

    def configure_optimizers(self):
        no_decay_keywords = ("bias", "LayerNorm", "layernorm", "layer_norm", "ln")
        feature_head_keyword = ("feature_gmae_to_gaussians", "magnitude_head")

        decay_params = []
        no_decay_params = []
        feature_head_decay_params = []
        feature_head_no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_feature_head = any(kw in name for kw in feature_head_keyword)
            is_no_decay = any(kw in name for kw in no_decay_keywords)

            if is_feature_head:
                if is_no_decay:
                    feature_head_no_decay_params.append(param)
                else:
                    feature_head_decay_params.append(param)
            else:
                if is_no_decay:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.optimizer_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
            {
                "params": feature_head_decay_params,
                "weight_decay": self.optimizer_cfg.feature_head_weight_decay,
            },
            {"params": feature_head_no_decay_params, "weight_decay": 0.0},
        ]
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        optimizer = torch.optim.AdamW(
            param_groups, lr=self.optimizer_cfg.lr, betas=(0.9, 0.95)
        )

        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer, 1 / warm_up_steps, 1, total_iters=warm_up_steps
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=get_cfg()["trainer"]["max_steps"],
            eta_min=self.optimizer_cfg.lr * 0.1,
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warm_up, lr_scheduler], milestones=[warm_up_steps]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
