#!/usr/bin/env python3
"""Modal runner to score exported SAM / C3G-SAM masks against GT test labels.

Examples::

    modal run src/modal/get_scores.py --experiment sam --wait
    modal run src/modal/get_scores.py --experiment c3gsam --wait
    modal run src/modal/get_scores.py --experiment c3gsam_ema-mag-uproj --wait
    modal run src/modal/get_scores.py::smoke --experiment sam --wait

Local equivalent::

    python -m src.evaluation.score_masks --experiment c3gsam
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from src.evaluation.score_masks import (
    DEFAULT_DILATION_RATIO,
    DEFAULT_MIN_OBJECT_PIXELS,
    DEFAULT_NUM_WORKERS,
    SCORES_FILENAME,
    run_scoring,
    run_smoke_scoring,
)
from src.modal.common import (
    C3G_SAM_EMA_EVAL_OUTPUT_MOUNT,
    C3G_SAM_EMA_EVAL_OUTPUT_VOLUME,
    C3G_SAM_EMA_MAG_UPROJ_EVAL_OUTPUT_MOUNT,
    C3G_SAM_EMA_MAG_UPROJ_EVAL_OUTPUT_VOLUME,
    C3G_SAM_EVAL_OUTPUT_MOUNT,
    C3G_SAM_EVAL_OUTPUT_VOLUME,
    C3G_SAM_NOEMA_NOMAG_EVAL_OUTPUT_MOUNT,
    C3G_SAM_NOEMA_NOMAG_EVAL_OUTPUT_VOLUME,
    EXPERIMENT_PRED_ROOTS,
    ExperimentName,
    REPLICA_MOUNT,
    REPLICA_VOLUME,
    SCANNET_MOUNT,
    SCANNET_VOLUME,
    VANILLA_SAM_OUTPUT_MOUNT,
    VANILLA_SAM_OUTPUT_VOLUME,
    build_eval_sam_modal_image,
    resolve_dataset_root,
    resolve_detach,
    resolve_experiment,
)

APP_NAME = "c3g-get-scores"

app = modal.App(APP_NAME)

replica_volume = modal.Volume.from_name(REPLICA_VOLUME, create_if_missing=True)
scannet_volume = modal.Volume.from_name(SCANNET_VOLUME, create_if_missing=True)
vanilla_output_volume = modal.Volume.from_name(
    VANILLA_SAM_OUTPUT_VOLUME, create_if_missing=True
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

scores_image = build_eval_sam_modal_image()

SCORE_VOLUMES = {
    str(REPLICA_MOUNT): replica_volume,
    str(SCANNET_MOUNT): scannet_volume,
    str(VANILLA_SAM_OUTPUT_MOUNT): vanilla_output_volume,
    str(C3G_SAM_EVAL_OUTPUT_MOUNT): c3g_eval_output_volume,
    str(C3G_SAM_EMA_MAG_UPROJ_EVAL_OUTPUT_MOUNT): c3g_ema_mag_uproj_eval_output_volume,
    str(C3G_SAM_EMA_EVAL_OUTPUT_MOUNT): c3g_ema_eval_output_volume,
    str(C3G_SAM_NOEMA_NOMAG_EVAL_OUTPUT_MOUNT): c3g_noema_nomag_eval_output_volume,
}


def _commit_pred_volume(experiment: ExperimentName) -> None:
    if experiment == "sam":
        vanilla_output_volume.commit()
    elif experiment == "c3gsam_ema-mag-uproj":
        c3g_ema_mag_uproj_eval_output_volume.commit()
    elif experiment == "c3gsam_ema":
        c3g_ema_eval_output_volume.commit()
    elif experiment == "c3gsam_noema-nomag":
        c3g_noema_nomag_eval_output_volume.commit()
    else:
        c3g_eval_output_volume.commit()


@app.function(
    image=scores_image,
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 6,
    volumes=SCORE_VOLUMES,
    nonpreemptible=True,
)
def compute_scores(
    experiment: str = "sam",
    replica_root: str | None = None,
    scannet_root: str | None = None,
    pred_root: str | None = None,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
    dilation_ratio: float = DEFAULT_DILATION_RATIO,
    output_path: str | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> dict:
    """Score exported masks on Modal worker volumes."""
    experiment_name = resolve_experiment(experiment)
    replica_data_root = resolve_dataset_root("replica", replica_root)
    scannet_data_root = resolve_dataset_root("scannet", scannet_root)
    predictions_root = (
        Path(pred_root) if pred_root else EXPERIMENT_PRED_ROOTS[experiment_name]
    )

    for dataset_name, root in (
        ("replica", replica_data_root),
        ("scannet", scannet_data_root),
    ):
        if not Path(root).is_dir():
            raise FileNotFoundError(
                f"{dataset_name} dataset not found at {root}. "
                f"Populate the `{dataset_name}` volume."
            )
    if not predictions_root.is_dir():
        raise FileNotFoundError(
            f"Predictions not found at {predictions_root}. "
            f"Run mask export for experiment={experiment_name!r} first."
        )

    report = run_scoring(
        experiment=experiment_name,
        replica_root=replica_data_root,
        scannet_root=scannet_data_root,
        pred_root=predictions_root,
        min_object_pixels=min_object_pixels,
        dilation_ratio=dilation_ratio,
        num_workers=num_workers,
    )

    out_path = Path(output_path) if output_path else predictions_root / SCORES_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _commit_pred_volume(experiment_name)
    print(f"Wrote scores to {out_path}")
    return report


@app.function(
    image=scores_image,
    cpu=8,
    memory=8192,
    timeout=60 * 30,
    volumes=SCORE_VOLUMES,
    nonpreemptible=True,
)
def smoke_scores(
    experiment: str = "sam",
    dataset: str = "replica",
    dataset_root: str | None = None,
    pred_root: str | None = None,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
    num_workers: int = 8,
) -> dict:
    """Score one test scene as a quick sanity check."""
    experiment_name = resolve_experiment(experiment)
    data_root = resolve_dataset_root(dataset, dataset_root)  # type: ignore[arg-type]
    predictions_root = (
        Path(pred_root) if pred_root else EXPERIMENT_PRED_ROOTS[experiment_name]
    )
    return run_smoke_scoring(
        experiment=experiment_name,
        dataset=dataset,
        dataset_root=data_root,
        pred_root=predictions_root,
        min_object_pixels=min_object_pixels,
        num_workers=num_workers,
    )


def _dispatch(fn, *, job_name: str, detach: bool, **kwargs) -> None:
    from src.misc.modal_run import dispatch_remote

    dispatch_remote(
        fn,
        detach=detach,
        job_name=job_name,
        app_name=APP_NAME,
        **kwargs,
    )


@app.local_entrypoint()
def main(
    experiment: str = "sam",
    replica_root: str | None = None,
    scannet_root: str | None = None,
    pred_root: str | None = None,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
    dilation_ratio: float = DEFAULT_DILATION_RATIO,
    output_path: str | None = None,
    detach: bool | None = None,
    wait: bool = False,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> None:
    """Score exported masks for Replica + ScanNet test splits."""
    resolve_experiment(experiment)
    _dispatch(
        compute_scores,
        job_name=f"{experiment} mask scoring",
        detach=resolve_detach(detach=detach, remote_job=not wait),
        experiment=experiment,
        replica_root=replica_root,
        scannet_root=scannet_root,
        pred_root=pred_root,
        min_object_pixels=min_object_pixels,
        dilation_ratio=dilation_ratio,
        output_path=output_path,
        num_workers=num_workers,
    )


@app.local_entrypoint()
def smoke(
    experiment: str = "sam",
    dataset: str = "replica",
    dataset_root: str | None = None,
    pred_root: str | None = None,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
    detach: bool | None = None,
    wait: bool = False,
    num_workers: int = 8,
) -> None:
    """Smoke test: score a single test scene."""
    resolve_experiment(experiment)
    _dispatch(
        smoke_scores,
        job_name=f"{experiment} mask scoring smoke ({dataset})",
        detach=resolve_detach(detach=detach, remote_job=not wait),
        experiment=experiment,
        dataset=dataset,
        dataset_root=dataset_root,
        pred_root=pred_root,
        min_object_pixels=min_object_pixels,
        num_workers=num_workers,
    )
