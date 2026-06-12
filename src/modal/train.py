#!/usr/bin/env python3
"""Modal runner for ScanNet SAM training (distillation or prompted).

Prerequisites (once)::

    modal volume put c3g-weights /path/to/gaussian_decoder.ckpt gaussian_decoder.ckpt
    modal volume put c3g-weights /path/to/sam_vit_h.pth sam_vit_h.pth
    # ScanNet frames on ``scannet`` volume.
    # Precomputed ``*_sam.pt`` on ``precompute_sam_features`` at ``scannet/`` speed up
    # distillation (required) and prompted training (optional; see
    # dataset.scannet_2dseg.sam_features_root).

Training (Hydra ``+training=`` only; pick experiment via CLI)::

    modal run src/modal/train.py
    modal run src/modal/train.py --experiment prompted --wait

Smoke (one optimizer step)::

    modal run src/modal/train.py::smoke --wait
    modal run src/modal/train.py::smoke --experiment prompted --wait

Checkpoints: ``c3g-train-outputs`` volume at ``/outputs/runs/<wandb.name>/``.
Prompted training saves checkpoints from ``val/loss`` (see
``feature_head_sam_prompted_scannet`` checkpointing config), not IoU or train step.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Literal

import modal

from src.modal.common import (
    C3G_MODAL_WORKSPACE,
    OUTPUT_MOUNT,
    OUTPUT_VOLUME,
    PRECOMPUTE_SAM_FEATURES_MOUNT,
    PRECOMPUTE_SAM_FEATURES_VOLUME,
    SCANNET_MOUNT,
    SCANNET_VOLUME,
    WEIGHTS_MOUNT,
    WEIGHTS_VOLUME,
    build_c3g_modal_image,
    resolve_detach,
)

ExperimentKind = Literal["distillation", "prompted"]

TRAINING_CONFIG_BY_EXPERIMENT: dict[ExperimentKind, str] = {
    "distillation": "feature_head_sam_precomputed",
    "prompted": "feature_head_sam_prompted_scannet",
}
SMOKE_TRAINING_CONFIG_BY_EXPERIMENT: dict[ExperimentKind, str] = {
    "distillation": "feature_head_sam_precomputed_smoke",
    "prompted": "feature_head_sam_prompted_scannet_smoke",
}
FULL_TRAINING_CONFIGS = frozenset(TRAINING_CONFIG_BY_EXPERIMENT.values())

APP_NAME = "c3g-sam-precomputed-train"
WANDB_SECRET = modal.Secret.from_name("wandb")

# A100 training: CUDA 12.4 + PyTorch cu124, targeting Ampere sm_80.
TRAIN_CUDA_IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"
TRAIN_TORCH_CUDA_ARCH_LIST = "8.0"

train_image = build_c3g_modal_image(
    cuda_image=TRAIN_CUDA_IMAGE,
    torch_cuda_arch_list=TRAIN_TORCH_CUDA_ARCH_LIST,
)

app = modal.App(APP_NAME)
weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
scannet_volume = modal.Volume.from_name(SCANNET_VOLUME, create_if_missing=True)
precompute_volume = modal.Volume.from_name(
    PRECOMPUTE_SAM_FEATURES_VOLUME, create_if_missing=True
)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True)

VOLUMES = {
    str(WEIGHTS_MOUNT): weights_volume,
    str(SCANNET_MOUNT): scannet_volume,
    str(PRECOMPUTE_SAM_FEATURES_MOUNT): precompute_volume,
    str(OUTPUT_MOUNT): output_volume,
}


def resolve_training_config(
    experiment: str, *, smoke: bool = False
) -> str:
    if experiment not in TRAINING_CONFIG_BY_EXPERIMENT:
        choices = ", ".join(TRAINING_CONFIG_BY_EXPERIMENT)
        print(
            f"Unknown experiment {experiment!r}; choose one of: {choices}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    mapping = (
        SMOKE_TRAINING_CONFIG_BY_EXPERIMENT
        if smoke
        else TRAINING_CONFIG_BY_EXPERIMENT
    )
    return mapping[experiment]  # type: ignore[index]


@app.function(
    image=train_image,
    gpu="A100-40GB",
    cpu=8,
    timeout=60 * 60 * 24,
    memory=131072, #128 GB RAM
    secrets=[WANDB_SECRET],
    volumes=VOLUMES,
)
def train_sam(training_config: str) -> str:
    """Run ``src.main`` with ``+training=<config>`` (YAML-only, no CLI overrides)."""
    if training_config in FULL_TRAINING_CONFIGS and not os.environ.get(
        "WANDB_API_KEY"
    ):
        raise RuntimeError(
            f"{training_config} sets wandb.mode=online. "
            "Create a Modal secret: modal secret create wandb WANDB_API_KEY=<key> "
            "Or use the smoke entrypoint (wandb disabled)."
        )

    cmd = ["python", "-m", "src.main", f"+training={training_config}"]
    print("Running:", " ".join(cmd))
    print(f"Outputs on volume `{OUTPUT_VOLUME}` under {OUTPUT_MOUNT / 'runs'}")
    subprocess.run(cmd, check=True, cwd=str(C3G_MODAL_WORKSPACE))
    output_volume.commit()
    return str(OUTPUT_MOUNT / "runs")


@app.local_entrypoint()
def main(
    experiment: ExperimentKind = "distillation",
    detach: bool | None = None,
    wait: bool = False,
) -> None:
    """Full ScanNet SAM training for ``distillation`` or ``prompted``."""
    from src.misc.modal_run import dispatch_remote

    training_config = resolve_training_config(experiment, smoke=False)
    job_labels = {
        "distillation": "C3G SAM ScanNet distill train",
        "prompted": "C3G SAM ScanNet prompted train",
    }
    dispatch_remote(
        train_sam,
        training_config=training_config,
        detach=resolve_detach(detach=detach, remote_job=not wait),
        job_name=job_labels[experiment],
        app_name=APP_NAME,
    )


@app.local_entrypoint()
def smoke(
    experiment: ExperimentKind = "distillation",
    detach: bool | None = None,
    wait: bool = False,
) -> None:
    """One optimizer-step smoke test for ``distillation`` or ``prompted``."""
    from src.misc.modal_run import dispatch_remote

    training_config = resolve_training_config(experiment, smoke=True)
    job_labels = {
        "distillation": "C3G SAM ScanNet distill smoke",
        "prompted": "C3G SAM ScanNet prompted smoke",
    }
    dispatch_remote(
        train_sam,
        training_config=training_config,
        detach=resolve_detach(detach=detach, remote_job=not wait),
        job_name=job_labels[experiment],
        app_name=APP_NAME,
    )
