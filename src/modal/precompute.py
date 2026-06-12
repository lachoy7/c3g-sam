#!/usr/bin/env python3
"""Modal runner for pre-computing SAM ViT-H encoder features.

Reads RGB frames from the ``replica`` or ``scannet`` dataset volumes and
writes ``{frame_id}_sam.pt`` files to the ``precompute_sam_features`` volume
under ``replica/`` or ``scannet/``.

Upload SAM weights (once)::

    modal volume put c3g-weights /path/to/sam_vit_h.pth sam_vit_h.pth

Smoke test (first scene on the volume)::

    modal run src/modal/precompute.py::smoke --dataset replica
    modal run src/modal/precompute.py::smoke --dataset scannet --wait

Full precompute (detached by default; pass ``--wait`` to block locally)::

    modal run --detach src/modal/precompute.py::main --dataset replica
    modal run --detach src/modal/precompute.py::main --dataset scannet
    modal run src/modal/precompute.py::main --dataset scannet --wait
    modal run src/modal/precompute.py::main --dataset replica \\
        --scenes office0,office1 --overwrite --detach

Output layout on volume ``precompute_sam_features``::

    replica/<scene_id>/{frame_id}_sam.pt
    scannet/<scene_id>/{frame_id}_sam.pt
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import modal

from src.modal.common import (
    C3G_MODAL_WORKSPACE,
    DATASET_SPECS,
    PRECOMPUTE_SAM_FEATURES_MOUNT,
    PRECOMPUTE_SAM_FEATURES_VOLUME,
    REPLICA_MOUNT,
    REPLICA_VOLUME,
    SCANNET_MOUNT,
    SCANNET_VOLUME,
    WEIGHTS_MOUNT,
    WEIGHTS_VOLUME,
    DatasetName,
    build_c3g_modal_image,
    find_smoke_scene,
    resolve_dataset_root,
    resolve_sam_checkpoint,
    run_subprocess_with_precompute_commit,
)

APP_NAME = "c3g-precompute-sam-features"
WORKSPACE = C3G_MODAL_WORKSPACE

# H200 precompute: CUDA 12.4 + PyTorch cu124, targeting Hopper sm_90.
PRECOMPUTE_CUDA_IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"
PRECOMPUTE_TORCH_CUDA_ARCH_LIST = "9.0"

precompute_image = build_c3g_modal_image(
    cuda_image=PRECOMPUTE_CUDA_IMAGE,
    torch_cuda_arch_list=PRECOMPUTE_TORCH_CUDA_ARCH_LIST,
)

app = modal.App(APP_NAME)
weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
replica_volume = modal.Volume.from_name(REPLICA_VOLUME, create_if_missing=True)
scannet_volume = modal.Volume.from_name(SCANNET_VOLUME, create_if_missing=True)
precompute_volume = modal.Volume.from_name(
    PRECOMPUTE_SAM_FEATURES_VOLUME, create_if_missing=True
)


@app.function(
    image=precompute_image,
    gpu="H200",
    timeout=60 * 60 * 24,
    volumes={
        str(WEIGHTS_MOUNT): weights_volume,
        str(REPLICA_MOUNT): replica_volume,
        str(SCANNET_MOUNT): scannet_volume,
        str(PRECOMPUTE_SAM_FEATURES_MOUNT): precompute_volume,
    },
)
def precompute_sam_features(
    dataset: DatasetName = "replica",
    scenes: list[str] | None = None,
    batch_size: int = 32,
    overwrite: bool = False,
    dataset_root: str | None = None,
    sam_checkpoint: str | None = None,
    smoke: bool = False,
) -> str:
    """Run ``scripts/precompute_sam_features.py`` on a Modal GPU."""
    spec = DATASET_SPECS[dataset]
    input_root = resolve_dataset_root(dataset, dataset_root)
    if not Path(input_root).is_dir():
        raise FileNotFoundError(
            f"{spec['label']} dataset not found at {input_root}. "
            f"Populate the `{spec['volume']}` volume via the download script."
        )

    output_root = PRECOMPUTE_SAM_FEATURES_MOUNT / dataset
    output_root.mkdir(parents=True, exist_ok=True)

    sam_path = resolve_sam_checkpoint(sam_checkpoint)

    selected_scenes = list(scenes) if scenes else None
    if smoke:
        smoke_scene = find_smoke_scene(input_root, scenes=list(DATASET_SPECS[dataset]["scenes"]))  # type: ignore[arg-type]
        selected_scenes = [smoke_scene]
        print(f"Smoke scene: {smoke_scene}")

    cmd = [
        "python",
        "scripts/precompute_sam_features.py",
        "--dataset-root",
        input_root,
        "--output-root",
        str(output_root),
        "--dataset",
        dataset,
        "--sam-checkpoint",
        str(sam_path),
        "--batch-size",
        str(batch_size),
    ]
    if selected_scenes:
        cmd.extend(["--scenes", *selected_scenes])
    if overwrite:
        cmd.append("--overwrite")

    print("Running:", " ".join(shlex.quote(part) for part in cmd))
    print(
        f"Features on volume `{PRECOMPUTE_SAM_FEATURES_VOLUME}`: "
        f"{output_root}/<scene>/{{frame_id}}_sam.pt"
    )
    run_subprocess_with_precompute_commit(
        cmd=cmd,
        cwd=WORKSPACE,
        output_root=output_root,
        commit=precompute_volume.commit,
    )
    return str(output_root)


def _dispatch_precompute(
    *,
    dataset: DatasetName,
    scenes: list[str] | None,
    batch_size: int,
    overwrite: bool,
    dataset_root: str | None,
    sam_checkpoint: str | None,
    smoke: bool,
    detach: bool,
) -> None:
    from src.misc.modal_run import dispatch_remote

    mode_label = "smoke precompute" if smoke else "precompute"
    result = dispatch_remote(
        precompute_sam_features,
        dataset=dataset,
        scenes=scenes,
        batch_size=batch_size,
        overwrite=overwrite,
        dataset_root=dataset_root,
        sam_checkpoint=sam_checkpoint,
        smoke=smoke,
        detach=detach,
        job_name=f"C3G SAM {mode_label} ({dataset})",
        app_name=APP_NAME,
    )
    if detach:
        return
    print(f"Remote run finished: {result}")


@app.local_entrypoint()
def main(
    dataset: DatasetName = "replica",
    scenes: str = "",
    batch_size: int = 32,
    overwrite: bool = False,
    dataset_root: str | None = None,
    sam_checkpoint: str | None = None,
    detach: bool = True,
    wait: bool = False,
) -> None:
    """Pre-compute SAM encoder features for Replica or ScanNet."""
    if dataset not in DATASET_SPECS:
        print(
            f"Unknown dataset {dataset!r}; choose one of: {', '.join(DATASET_SPECS)}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    scene_list = [scene.strip() for scene in scenes.split(",") if scene.strip()] or None
    _dispatch_precompute(
        dataset=dataset,
        scenes=scene_list,
        batch_size=batch_size,
        overwrite=overwrite,
        dataset_root=dataset_root,
        sam_checkpoint=sam_checkpoint,
        smoke=False,
        detach=detach and not wait,
    )


@app.local_entrypoint()
def smoke(
    dataset: DatasetName = "replica",
    batch_size: int = 32,
    overwrite: bool = False,
    dataset_root: str | None = None,
    sam_checkpoint: str | None = None,
    detach: bool = True,
    wait: bool = False,
) -> None:
    """Pre-compute SAM features for the first scene on the dataset volume."""
    if dataset not in DATASET_SPECS:
        print(
            f"Unknown dataset {dataset!r}; choose one of: {', '.join(DATASET_SPECS)}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    _dispatch_precompute(
        dataset=dataset,
        scenes=None,
        batch_size=batch_size,
        overwrite=overwrite,
        dataset_root=dataset_root,
        sam_checkpoint=sam_checkpoint,
        smoke=True,
        detach=detach and not wait,
    )
