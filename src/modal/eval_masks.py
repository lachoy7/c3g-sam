#!/usr/bin/env python3
"""Modal mask export for vanilla SAM and C3G-SAM (shared layout and resume logic).

Per-class exports include ``<class_id>.png`` and ``<class_id>_logits.npy`` (full-res
logits before thresholding) for logit-aware overlap resolution at scoring time.

Heavy deps (torch, Hydra, C3G stack) are imported only inside remote workers, not
when Modal loads this file locally for ``modal run``.

Vanilla SAM (Replica + ScanNet test)::

    modal run src/modal/eval_masks.py::sam --wait
    modal deploy src/modal/eval_masks.py  # HTTP /predict on VanillaSAMService

C3G-SAM distillation::

    modal run src/modal/eval_masks.py::c3g --wait
    modal run --detach src/modal/eval_masks.py::c3g

C3G-SAM ablations::

    modal run src/modal/eval_masks.py::c3gsam_ema-mag-uproj --wait
    modal run src/modal/eval_masks.py::c3gsam_ema --wait
    modal run src/modal/eval_masks.py::c3gsam_noema-nomag --wait

Smoke::

    modal run src/modal/eval_masks.py::sam_smoke --dataset replica --wait
    modal run src/modal/eval_masks.py::c3g_smoke --wait

Local C3G export (no Modal)::

    python -m src.evaluation.mask_export \\
        +evaluation=c3g_sam_distill checkpointing.load=/path/to.ckpt
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal

from src.evaluation.eval_common import (
    C3G_EVAL_DATASETS,
    VANILLA_EVAL_DATASETS,
)
from src.modal.common import (
    C3G_MODAL_WORKSPACE,
    C3G_SAM_EMA_EVAL_OUTPUT_MOUNT,
    C3G_SAM_EMA_EVAL_OUTPUT_VOLUME,
    C3G_SAM_EMA_MAG_UPROJ_EVAL_OUTPUT_MOUNT,
    C3G_SAM_EMA_MAG_UPROJ_EVAL_OUTPUT_VOLUME,
    C3G_SAM_EVAL_OUTPUT_MOUNT,
    C3G_SAM_EVAL_OUTPUT_VOLUME,
    C3G_SAM_NOEMA_NOMAG_EVAL_OUTPUT_MOUNT,
    C3G_SAM_NOEMA_NOMAG_EVAL_OUTPUT_VOLUME,
    DATASET_SPECS,
    PRECOMPUTE_SAM_FEATURES_MOUNT,
    PRECOMPUTE_SAM_FEATURES_VOLUME,
    REPLICA_MOUNT,
    REPLICA_VOLUME,
    SCANNET_MOUNT,
    SCANNET_VOLUME,
    VANILLA_SAM_OUTPUT_MOUNT,
    VANILLA_SAM_OUTPUT_VOLUME,
    WEIGHTS_MOUNT,
    WEIGHTS_VOLUME,
    DatasetName,
    build_c3g_modal_image,
    build_eval_sam_modal_image,
    find_modal_repo_root,
    find_smoke_frame,
    resolve_dataset_root,
    resolve_detach,
    resolve_distillation_checkpoint,
    resolve_sam_checkpoint,
    sample_eval_visualization_keys,
)

APP_NAME = "c3g-mask-eval"
WANDB_SECRET = modal.Secret.from_name("wandb")

DEFAULT_SAM_VARIANT = "sam_vit_h"
DEFAULT_BATCH_SIZE = 32
DEFAULT_CHECKPOINT = "distillation-base.ckpt"
DFT_CHECKPOINT = "distillation-diff_learnable_tokens.ckpt"
NOMAGHEAD_CHECKPOINT = "c3gsam-nomaghead.ckpt"
EMA_NOMAG_CHECKPOINT = "ema-nomag.ckpt"
C3G_EVAL_CONFIG = "c3g_sam_distill"
DEFAULT_PROMPT_STRATEGY = "centroid"
DEFAULT_MIN_OBJECT_PIXELS = 16
DEFAULT_VISUALIZATION_COUNT = 5
DEFAULT_VISUALIZATION_SEED = 42

# Vanilla SAM: H200 + Hopper sm_90.
VANILLA_TORCH_CUDA_ARCH_LIST = "9.0"
# C3G eval: T4 + Turing sm_75.
C3G_EVAL_CUDA_IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"
C3G_EVAL_TORCH_CUDA_ARCH_LIST = "7.5"

app = modal.App(APP_NAME)

weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
replica_volume = modal.Volume.from_name(REPLICA_VOLUME, create_if_missing=True)
scannet_volume = modal.Volume.from_name(SCANNET_VOLUME, create_if_missing=True)
vanilla_output_volume = modal.Volume.from_name(
    VANILLA_SAM_OUTPUT_VOLUME, create_if_missing=True
)
precompute_volume = modal.Volume.from_name(
    PRECOMPUTE_SAM_FEATURES_VOLUME, create_if_missing=True
)
c3g_eval_output_volume = modal.Volume.from_name(
    C3G_SAM_EVAL_OUTPUT_VOLUME, create_if_missing=True
)
c3g_ema_mag_uproj_eval_output_volume = modal.Volume.from_name(
    C3G_SAM_EMA_MAG_UPROJ_EVAL_OUTPUT_VOLUME, create_if_missing=True
)
c3g_ema_eval_output_volume = modal.Volume.from_name(
    C3G_SAM_EMA_EVAL_OUTPUT_VOLUME, create_if_missing=True
)
c3g_noema_nomag_eval_output_volume = modal.Volume.from_name(
    C3G_SAM_NOEMA_NOMAG_EVAL_OUTPUT_VOLUME, create_if_missing=True
)

vanilla_eval_image = build_eval_sam_modal_image(
    extra_env={"TORCH_CUDA_ARCH_LIST": VANILLA_TORCH_CUDA_ARCH_LIST}
)
_repo_root = find_modal_repo_root()
if _repo_root is not None:
    c3g_eval_image = build_c3g_modal_image(
        cuda_image=C3G_EVAL_CUDA_IMAGE,
        torch_cuda_arch_list=C3G_EVAL_TORCH_CUDA_ARCH_LIST,
        repo_root=_repo_root,
    )
else:
    # Vanilla workers mount only /root/src; C3G image is baked at local deploy time.
    c3g_eval_image = vanilla_eval_image

C3G_VOLUMES = {
    str(WEIGHTS_MOUNT): weights_volume,
    str(REPLICA_MOUNT): replica_volume,
    str(SCANNET_MOUNT): scannet_volume,
    str(PRECOMPUTE_SAM_FEATURES_MOUNT): precompute_volume,
    str(C3G_SAM_EVAL_OUTPUT_MOUNT): c3g_eval_output_volume,
    str(C3G_SAM_EMA_MAG_UPROJ_EVAL_OUTPUT_MOUNT): c3g_ema_mag_uproj_eval_output_volume,
    str(C3G_SAM_EMA_EVAL_OUTPUT_MOUNT): c3g_ema_eval_output_volume,
    str(C3G_SAM_NOEMA_NOMAG_EVAL_OUTPUT_MOUNT): c3g_noema_nomag_eval_output_volume,
}
_EVAL_OUTPUT_VOLUMES = {
    str(C3G_SAM_EVAL_OUTPUT_MOUNT): c3g_eval_output_volume,
    str(C3G_SAM_EMA_MAG_UPROJ_EVAL_OUTPUT_MOUNT): c3g_ema_mag_uproj_eval_output_volume,
    str(C3G_SAM_EMA_EVAL_OUTPUT_MOUNT): c3g_ema_eval_output_volume,
    str(C3G_SAM_NOEMA_NOMAG_EVAL_OUTPUT_MOUNT): c3g_noema_nomag_eval_output_volume,
}
_EVAL_OUTPUT_VOLUME_NAMES = {
    str(C3G_SAM_EVAL_OUTPUT_MOUNT): C3G_SAM_EVAL_OUTPUT_VOLUME,
    str(C3G_SAM_EMA_MAG_UPROJ_EVAL_OUTPUT_MOUNT): C3G_SAM_EMA_MAG_UPROJ_EVAL_OUTPUT_VOLUME,
    str(C3G_SAM_EMA_EVAL_OUTPUT_MOUNT): C3G_SAM_EMA_EVAL_OUTPUT_VOLUME,
    str(C3G_SAM_NOEMA_NOMAG_EVAL_OUTPUT_MOUNT): C3G_SAM_NOEMA_NOMAG_EVAL_OUTPUT_VOLUME,
}


@app.cls(
    image=vanilla_eval_image,
    gpu="H200",
    volumes={
        str(WEIGHTS_MOUNT): weights_volume,
        str(REPLICA_MOUNT): replica_volume,
        str(SCANNET_MOUNT): scannet_volume,
        str(VANILLA_SAM_OUTPUT_MOUNT): vanilla_output_volume,
    },
    timeout=60 * 60 * 24,
    scaledown_window=300,
)
class VanillaSAMService:
    """Stateful vanilla SAM worker (HTTP + full mask export)."""

    @modal.enter()
    def load_model(self) -> None:
        import torch

        from src.model.sam import load_sam

        sam_path = resolve_sam_checkpoint()
        print(f"SAM checkpoint: {sam_path}")
        self.device = torch.device("cuda")
        self.sam = load_sam(DEFAULT_SAM_VARIANT, str(sam_path), freeze=True).to(
            self.device
        )
        self.sam.eval()

    @modal.method()
    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        from src.evaluation.mask_export import run_vanilla_sam_predict

        return run_vanilla_sam_predict(self.sam, self.device, payload)

    @modal.method()
    def predict_smoke_frame(
        self,
        dataset: DatasetName = "replica",
        dataset_root: str | None = None,
        prompt_strategy: str = DEFAULT_PROMPT_STRATEGY,
        min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
    ) -> dict[str, Any]:
        from src.evaluation.mask_export import (
            build_vanilla_batch_predict_payload,
            run_vanilla_sam_predict,
        )

        spec = DATASET_SPECS[dataset]
        root = resolve_dataset_root(dataset, dataset_root)
        scene_id, paths = find_smoke_frame(root, scenes=list(spec["scenes"]))  # type: ignore[arg-type]
        summary = f"{dataset} scene={scene_id} frame={paths.frame_id} ({paths.image.name})"
        print(f"Smoke test image: {summary}")

        payload = build_vanilla_batch_predict_payload(
            [(paths.image.read_bytes(), paths.label)],
            prompt_strategy=prompt_strategy,
            min_object_pixels=min_object_pixels,
        )
        result = run_vanilla_sam_predict(self.sam, self.device, payload)
        result["smoke_frame"] = summary
        result["prompt_strategy"] = prompt_strategy
        return result

    @modal.method()
    def predict_local_files(
        self,
        image_bytes: bytes,
        label_bytes: bytes,
        prompt_strategy: str = DEFAULT_PROMPT_STRATEGY,
        min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
    ) -> dict[str, Any]:
        """Predict from raw image/label bytes (CLI uploads local files to Modal)."""
        from src.evaluation.mask_export import (
            build_vanilla_batch_predict_payload,
            run_vanilla_sam_predict,
        )

        label_path = Path("/tmp/vanilla_eval_label.png")
        label_path.write_bytes(label_bytes)
        payload = build_vanilla_batch_predict_payload(
            [(image_bytes, label_path)],
            prompt_strategy=prompt_strategy,
            min_object_pixels=min_object_pixels,
        )
        return run_vanilla_sam_predict(self.sam, self.device, payload)

    @modal.method()
    def export_masks(
        self,
        replica_root: str | None = None,
        scannet_root: str | None = None,
        prompt_strategy: str = DEFAULT_PROMPT_STRATEGY,
        min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> str:
        from src.evaluation.mask_export import export_vanilla_sam_masks

        dataset_roots = {
            "replica": resolve_dataset_root("replica", replica_root),
            "scannet": resolve_dataset_root("scannet", scannet_root),
        }
        for dataset_name, data_root in dataset_roots.items():
            spec = DATASET_SPECS[dataset_name]  # type: ignore[index]
            if not Path(data_root).is_dir():
                raise FileNotFoundError(
                    f"{spec['label']} dataset not found at {data_root}. "
                    f"Populate the `{spec['volume']}` volume."
                )

        export_vanilla_sam_masks(
            self.sam,
            self.device,
            output_root=VANILLA_SAM_OUTPUT_MOUNT,
            dataset_roots=dataset_roots,
            prompt_strategy=prompt_strategy,
            min_object_pixels=min_object_pixels,
            batch_size=batch_size,
        )
        vanilla_output_volume.commit()
        print(f"Output volume: `{VANILLA_SAM_OUTPUT_VOLUME}`")
        return str(VANILLA_SAM_OUTPUT_MOUNT)

    @modal.fastapi_endpoint(method="POST", docs=True)
    def web(self):
        from fastapi import FastAPI
        from pydantic import BaseModel, Field

        class PredictBody(BaseModel):
            images_b64: list[str] = Field(...)
            point_coords: list[list[list[float]]] | None = None
            point_labels: list[list[int]] | None = None
            boxes: list[list[float]] | None = None
            multimask_output: bool = True
            return_logits: bool = False

        service = self
        api = FastAPI(title="C3G Vanilla SAM")

        @api.post("/predict")
        def predict_endpoint(body: PredictBody) -> dict[str, Any]:
            return service.predict.local(body.model_dump())

        return api


@app.function(
    image=c3g_eval_image,
    gpu="T4",
    cpu=8,
    timeout=60 * 60 * 24,
    memory=131072,
    secrets=[WANDB_SECRET],
    volumes=C3G_VOLUMES,
)
def export_c3g_masks(
    evaluation_config: str = C3G_EVAL_CONFIG,
    checkpoint_name: str = DEFAULT_CHECKPOINT,
    eval_output_mount: str = str(C3G_SAM_EVAL_OUTPUT_MOUNT),
    different_learnable_tokens: bool = False,
    mask_batch_size: int = DEFAULT_BATCH_SIZE,
    limit_frames: int | None = None,
    with_lightning_test: bool = False,
    visualization_count: int = DEFAULT_VISUALIZATION_COUNT,
    visualization_seed: int = DEFAULT_VISUALIZATION_SEED,
    wandb_mode: str = "online",
    limit_test_batches: int | None = None,
) -> str:
    """Export C3G-SAM masks (all ML imports run on the Modal worker)."""
    import torch
    from hydra import compose, initialize_config_dir

    from src.config import load_typed_root_config
    from src.dataset import get_dataset
    from src.evaluation.mask_export import (
        build_distillation_wrapper,
        export_c3g_sam_masks,
        load_checkpoint_into_wrapper,
    )
    from src.global_cfg import set_cfg
    from src.misc.step_tracker import StepTracker
    from src.model.sam_decoder import SAMMaskDecoderWrapper

    checkpoint_path = resolve_distillation_checkpoint(WEIGHTS_MOUNT / checkpoint_name)

    for dataset_name, _, _ in C3G_EVAL_DATASETS:
        spec = DATASET_SPECS[dataset_name]
        mount = REPLICA_MOUNT if dataset_name == "replica" else SCANNET_MOUNT
        if not mount.is_dir():
            raise FileNotFoundError(
                f"{spec['label']} not mounted at {mount}. "
                f"Populate `{spec['volume']}`."
            )

    print(f"Checkpoint: {checkpoint_path}")

    config_dir = str(C3G_MODAL_WORKSPACE / "config")
    overrides = [
        f"+evaluation={evaluation_config}",
        f"checkpointing.load={checkpoint_path}",
        f"eval.mask_batch_size={mask_batch_size}",
        f"eval.mask_output_dir={eval_output_mount}",
    ]
    if different_learnable_tokens:
        overrides.append("model.encoder.different_learnable_tokens=true")
    if limit_frames is not None:
        overrides.append(f"eval.limit_frames={limit_frames}")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg_dict = compose(config_name="main", overrides=overrides)

    set_cfg(cfg_dict)
    eval_cfg = cfg_dict.get("eval", {})
    device = torch.device("cuda")

    wrapper = build_distillation_wrapper(cfg_dict)
    load_checkpoint_into_wrapper(wrapper, checkpoint_path)
    wrapper.eval()
    wrapper.to(device)

    sam_ckpt = cfg_dict.debug_decoder.get(
        "sam_checkpoint",
        cfg_dict.train.get("sam_checkpoint", "/weights/sam_vit_h.pth"),
    )
    sam_variant = cfg_dict.debug_decoder.get(
        "sam_model_variant",
        cfg_dict.train.get("sam_model_variant", DEFAULT_SAM_VARIANT),
    )
    mask_decoder = SAMMaskDecoderWrapper(sam_ckpt, model_variant=sam_variant).to(device)
    mask_decoder.eval()

    typed_cfg = load_typed_root_config(cfg_dict)
    datasets = get_dataset(typed_cfg.dataset, "test", StepTracker())
    datasets_by_name = {ds.cfg.name: ds for ds in datasets}

    export_c3g_sam_masks(
        wrapper,
        mask_decoder,
        datasets_by_name,
        output_root=Path(eval_output_mount),
        cfg_dict=cfg_dict,
        prompt_strategy=eval_cfg.get("prompt_strategy", DEFAULT_PROMPT_STRATEGY),
        min_object_pixels=int(
            eval_cfg.get("min_object_pixels", DEFAULT_MIN_OBJECT_PIXELS)
        ),
        mask_batch_size=int(eval_cfg.get("mask_batch_size", mask_batch_size)),
        limit_frames=eval_cfg.get("limit_frames"),
    )

    if with_lightning_test:
        if wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("wandb.mode=online requires WANDB_API_KEY in Modal secret.")
        visualization_keys: list[str] = []
        per_dataset = max(1, visualization_count // len(C3G_EVAL_DATASETS))
        for dataset_name, _, scenes in C3G_EVAL_DATASETS:
            mount = REPLICA_MOUNT if dataset_name == "replica" else SCANNET_MOUNT
            visualization_keys.extend(
                sample_eval_visualization_keys(
                    mount,
                    list(scenes),
                    count=per_dataset,
                    seed=visualization_seed,
                )
            )
        visualization_keys = visualization_keys[:visualization_count]
        test_cmd = [
            "python",
            "-m",
            "src.main",
            f"+evaluation={evaluation_config}",
            f"checkpointing.load={checkpoint_path}",
            f"eval.visualization_keys={json.dumps(visualization_keys)}",
            f"wandb.mode={wandb_mode}",
            f"hydra.run.dir={eval_output_mount}/lightning",
        ]
        if limit_test_batches is not None:
            test_cmd.append(f"trainer.limit_test_batches={limit_test_batches}")
        if different_learnable_tokens:
            test_cmd.append("model.encoder.different_learnable_tokens=true")
        print("Running Lightning test:", " ".join(test_cmd))
        subprocess.run(test_cmd, check=True, cwd=str(C3G_MODAL_WORKSPACE))

    _EVAL_OUTPUT_VOLUMES[eval_output_mount].commit()
    print(f"Output volume: `{_EVAL_OUTPUT_VOLUME_NAMES[eval_output_mount]}`")
    return eval_output_mount


def _dispatch_vanilla(method_name: str, *, job_name: str, detach: bool, **kwargs) -> None:
    from src.misc.modal_run import dispatch_remote

    dispatch_remote(
        getattr(VanillaSAMService(), method_name),
        detach=detach,
        job_name=job_name,
        app_name=APP_NAME,
        **kwargs,
    )


@app.local_entrypoint(name="sam")
def sam(
    image_path: str | None = None,
    label_path: str | None = None,
    replica_root: str | None = None,
    scannet_root: str | None = None,
    prompt_strategy: str = DEFAULT_PROMPT_STRATEGY,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    detach: bool | None = None,
    wait: bool = False,
) -> None:
    """Full SAM mask export, or single-image predict (bytes sent to Modal)."""
    use_detach = resolve_detach(detach=detach, remote_job=not wait)

    if image_path:
        image_file = Path(image_path)
        if not image_file.is_file():
            print(f"Image not found: {image_file}", file=sys.stderr)
            raise SystemExit(1)
        if not label_path:
            print("--label-path is required.", file=sys.stderr)
            raise SystemExit(2)
        label_file = Path(label_path)
        if not label_file.is_file():
            print(f"Label not found: {label_file}", file=sys.stderr)
            raise SystemExit(1)
        _dispatch_vanilla(
            "predict_local_files",
            job_name="SAM predict",
            detach=use_detach,
            image_bytes=image_file.read_bytes(),
            label_bytes=label_file.read_bytes(),
            prompt_strategy=prompt_strategy,
            min_object_pixels=min_object_pixels,
        )
        return

    _dispatch_vanilla(
        "export_masks",
        job_name="SAM mask export",
        detach=use_detach,
        replica_root=replica_root,
        scannet_root=scannet_root,
        prompt_strategy=prompt_strategy,
        min_object_pixels=min_object_pixels,
        batch_size=batch_size,
    )


@app.local_entrypoint(name="sam_smoke")
def sam_smoke(
    dataset: DatasetName = "replica",
    dataset_root: str | None = None,
    prompt_strategy: str = DEFAULT_PROMPT_STRATEGY,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
    detach: bool | None = None,
    wait: bool = False,
) -> None:
    if dataset not in DATASET_SPECS:
        raise SystemExit(f"Unknown dataset {dataset!r}")
    _dispatch_vanilla(
        "predict_smoke_frame",
        job_name=f"SAM smoke ({dataset})",
        detach=resolve_detach(detach=detach, remote_job=not wait),
        dataset=dataset,
        dataset_root=dataset_root,
        prompt_strategy=prompt_strategy,
        min_object_pixels=min_object_pixels,
    )


@app.local_entrypoint()
def c3g(
    checkpoint_name: str = DEFAULT_CHECKPOINT,
    mask_batch_size: int = DEFAULT_BATCH_SIZE,
    with_lightning_test: bool = False,
    detach: bool | None = None,
    wait: bool = False,
) -> None:
    """Full C3G-SAM mask export on Replica + ScanNet test."""
    from src.misc.modal_run import dispatch_remote

    dispatch_remote(
        export_c3g_masks,
        checkpoint_name=checkpoint_name,
        mask_batch_size=mask_batch_size,
        with_lightning_test=with_lightning_test,
        detach=resolve_detach(detach=detach, remote_job=not wait),
        job_name="C3G-SAM mask export",
        app_name=APP_NAME,
    )


@app.local_entrypoint(name="c3gsam_ema-mag-uproj")
def c3gsam_ema_mag_uproj(
    mask_batch_size: int = DEFAULT_BATCH_SIZE,
    with_lightning_test: bool = False,
    detach: bool | None = None,
    wait: bool = False,
) -> None:
    """C3G-SAM mask export using distillation-diff_learnable_tokens.ckpt."""
    from src.misc.modal_run import dispatch_remote

    dispatch_remote(
        export_c3g_masks,
        checkpoint_name=DFT_CHECKPOINT,
        eval_output_mount=str(C3G_SAM_EMA_MAG_UPROJ_EVAL_OUTPUT_MOUNT),
        different_learnable_tokens=True,
        mask_batch_size=mask_batch_size,
        with_lightning_test=with_lightning_test,
        detach=resolve_detach(detach=detach, remote_job=not wait),
        job_name="C3G-SAM EMA mag up-proj mask export",
        app_name=APP_NAME,
    )


@app.local_entrypoint(name="c3gsam_ema")
def c3gsam_ema(
    mask_batch_size: int = DEFAULT_BATCH_SIZE,
    with_lightning_test: bool = False,
    detach: bool | None = None,
    wait: bool = False,
) -> None:
    """C3G-SAM mask export using c3gsam-nomaghead.ckpt (same config as ``c3g``)."""
    from src.misc.modal_run import dispatch_remote

    dispatch_remote(
        export_c3g_masks,
        checkpoint_name=NOMAGHEAD_CHECKPOINT,
        eval_output_mount=str(C3G_SAM_EMA_EVAL_OUTPUT_MOUNT),
        mask_batch_size=mask_batch_size,
        with_lightning_test=with_lightning_test,
        detach=resolve_detach(detach=detach, remote_job=not wait),
        job_name="C3G-SAM EMA mask export",
        app_name=APP_NAME,
    )


@app.local_entrypoint(name="c3gsam_noema-nomag")
def c3gsam_noema_nomag(
    mask_batch_size: int = DEFAULT_BATCH_SIZE,
    with_lightning_test: bool = False,
    detach: bool | None = None,
    wait: bool = False,
) -> None:
    """C3G-SAM mask export using ema-nomag.ckpt (same config as ``c3g``)."""
    from src.misc.modal_run import dispatch_remote

    dispatch_remote(
        export_c3g_masks,
        checkpoint_name=EMA_NOMAG_CHECKPOINT,
        eval_output_mount=str(C3G_SAM_NOEMA_NOMAG_EVAL_OUTPUT_MOUNT),
        mask_batch_size=mask_batch_size,
        with_lightning_test=with_lightning_test,
        detach=resolve_detach(detach=detach, remote_job=not wait),
        job_name="C3G-SAM no-EMA no-mag mask export",
        app_name=APP_NAME,
    )


@app.local_entrypoint()
def c3g_smoke(
    checkpoint_name: str = DEFAULT_CHECKPOINT,
    mask_batch_size: int = 8,
    detach: bool | None = None,
    wait: bool = False,
) -> None:
    from src.misc.modal_run import dispatch_remote

    dispatch_remote(
        export_c3g_masks,
        checkpoint_name=checkpoint_name,
        mask_batch_size=mask_batch_size,
        limit_frames=1,
        detach=resolve_detach(detach=detach, remote_job=not wait),
        job_name="C3G-SAM mask export smoke",
        app_name=APP_NAME,
    )
