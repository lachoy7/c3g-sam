"""Modal volume mounts, image builders, and dispatch helpers."""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

from src.evaluation.eval_common import (
    REPLICA_2DSEG_SCENES,
    SCANNET_2DSEG_SCENES,
    SCANNET_2DSEG_TEST_SCENES,
    DatasetName,
    ExperimentName,
    expected_mask_class_ids,
    find_smoke_frame,
    find_smoke_scene,
    iter_dataset_frames,
    resolve_experiment,
    sample_eval_visualization_keys,
)

WEIGHTS_VOLUME = "c3g-weights"
OUTPUT_VOLUME = "c3g-train-outputs"
VANILLA_SAM_OUTPUT_VOLUME = "vanilla-sam-outputs"
C3G_SAM_EVAL_OUTPUT_VOLUME = "c3g-sam-eval-outputs"
# Modal volume names kept for backward compatibility with existing deployments.
C3G_SAM_EMA_MAG_UPROJ_EVAL_OUTPUT_VOLUME = "c3g-sam-dft-eval-outputs"
C3G_SAM_EMA_EVAL_OUTPUT_VOLUME = "c3g-sam-nomaghead-eval-outputs"
C3G_SAM_NOEMA_NOMAG_EVAL_OUTPUT_VOLUME = "c3g-sam-ema-nomag-eval-outputs"
PRECOMPUTE_SAM_FEATURES_VOLUME = "precompute_sam_features"
REPLICA_VOLUME = "replica"
SCANNET_VOLUME = "scannet"

WEIGHTS_MOUNT = Path("/weights")
REPLICA_MOUNT = Path("/replica")
SCANNET_MOUNT = Path("/scannet")
OUTPUT_MOUNT = Path("/outputs")
VANILLA_SAM_OUTPUT_MOUNT = Path("/vanilla-sam-outputs")
C3G_SAM_EVAL_OUTPUT_MOUNT = Path("/c3g-sam-eval-outputs")
C3G_SAM_EMA_MAG_UPROJ_EVAL_OUTPUT_MOUNT = Path("/c3g-sam-dft-eval-outputs")
C3G_SAM_EMA_EVAL_OUTPUT_MOUNT = Path("/c3g-sam-nomaghead-eval-outputs")
C3G_SAM_NOEMA_NOMAG_EVAL_OUTPUT_MOUNT = Path("/c3g-sam-ema-nomag-eval-outputs")
PRECOMPUTE_SAM_FEATURES_MOUNT = Path("/precompute_sam_features")

EXPERIMENT_PRED_ROOTS: dict[ExperimentName, Path] = {
    "sam": VANILLA_SAM_OUTPUT_MOUNT,
    "c3gsam": C3G_SAM_EVAL_OUTPUT_MOUNT,
    "c3gsam_ema-mag-uproj": C3G_SAM_EMA_MAG_UPROJ_EVAL_OUTPUT_MOUNT,
    "c3gsam_ema": C3G_SAM_EMA_EVAL_OUTPUT_MOUNT,
    "c3gsam_noema-nomag": C3G_SAM_NOEMA_NOMAG_EVAL_OUTPUT_MOUNT,
}

DEFAULT_SAM_CHECKPOINT = WEIGHTS_MOUNT / "sam_vit_h.pth"
DEFAULT_DISTILLATION_CHECKPOINT = WEIGHTS_MOUNT / "distillation-base.ckpt"

DATASET_SPECS: dict[DatasetName, dict[str, str | list[str]]] = {
    "replica": {
        "default_root": str(REPLICA_MOUNT),
        "volume": REPLICA_VOLUME,
        "label": "Replica",
        "scenes": REPLICA_2DSEG_SCENES,
    },
    "scannet": {
        "default_root": str(SCANNET_MOUNT),
        "volume": SCANNET_VOLUME,
        "label": "ScanNet",
        "scenes": SCANNET_2DSEG_SCENES,
    },
}


def resolve_dataset_root(dataset: DatasetName, dataset_root: str | None) -> str:
    return dataset_root or DATASET_SPECS[dataset]["default_root"]  # type: ignore[index]


def resolve_detach(*, detach: bool | None, remote_job: bool) -> bool:
    """Smoke/remote jobs detach by default; pass ``detach=False`` or ``--wait`` to block."""
    if detach is not None:
        return detach
    return remote_job


def resolve_sam_checkpoint(override: str | Path | None = None) -> Path:
    """Return the SAM ViT-H checkpoint from the ``c3g-weights`` volume."""
    path = Path(override) if override is not None else DEFAULT_SAM_CHECKPOINT
    if not path.is_file():
        raise FileNotFoundError(
            f"SAM checkpoint not found at {path}. "
            f"Upload with:\n"
            f"  modal volume put {WEIGHTS_VOLUME} /path/to/sam_vit_h.pth sam_vit_h.pth"
        )
    return path


def resolve_distillation_checkpoint(override: str | Path | None = None) -> Path:
    """Return the distillation Lightning checkpoint from the ``c3g-weights`` volume."""
    path = (
        Path(override) if override is not None else DEFAULT_DISTILLATION_CHECKPOINT
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Distillation checkpoint not found at {path}. "
            f"Upload with:\n"
            f"  modal volume put {WEIGHTS_VOLUME} /path/to/distillation-base.ckpt "
            f"distillation-base.ckpt"
        )
    return path


def run_subprocess_with_precompute_commit(
    *,
    cmd: list[str],
    cwd: Path | str,
    output_root: Path,
    commit: Callable[[], None],
    poll_interval_s: float = 30.0,
) -> None:
    """Run precompute subprocess and persist ``*_sam.pt`` files to a Modal volume."""
    stop_event = threading.Event()
    seen_features: set[str] = set()

    def _commit_loop() -> None:
        while not stop_event.is_set():
            if output_root.is_dir():
                for feature_path in output_root.rglob("*_sam.pt"):
                    rel = str(feature_path.relative_to(output_root))
                    if rel not in seen_features:
                        seen_features.add(rel)
                        commit()
                        print(f"Committed precompute volume: {rel}")

            stop_event.wait(poll_interval_s)

    thread = threading.Thread(
        target=_commit_loop,
        name="modal-precompute-commit",
        daemon=True,
    )
    thread.start()
    try:
        subprocess.run(cmd, check=True, cwd=str(cwd))
    finally:
        stop_event.set()
        thread.join(timeout=poll_interval_s + 5)
        commit()
        print(f"Committed precompute volume (final): {output_root}")


C3G_MODAL_WORKSPACE = Path("/workspace")
C3G_MODAL_PYTHON = "3.12"
VANILLA_SAM_MODAL_ROOT = Path("/root")
VANILLA_SAM_PYTHON = "3.11"
PYTORCH_CU124_INDEX = "https://download.pytorch.org/whl/cu124"
PYTORCH_CU128_INDEX = "https://download.pytorch.org/whl/cu128"
MODAL_UV_PATH = "/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def find_modal_repo_root(*, start: Path | None = None) -> Path | None:
    """Return repo root when ``pyproject.toml`` + ``src/`` exist, else ``None``."""
    here = start or Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    workspace = C3G_MODAL_WORKSPACE
    if (workspace / "pyproject.toml").is_file() and (workspace / "src").is_dir():
        return workspace
    return None


def build_eval_sam_modal_image(
    *,
    src_root: Path | None = None,
    remote_root: Path = VANILLA_SAM_MODAL_ROOT,
    extra_env: dict[str, str] | None = None,
):
    """Lightweight image for SAM mask eval (``uv pip install``)."""
    import modal

    src = src_root or Path(__file__).resolve().parent.parent
    image_env = {"PYTHONPATH": str(remote_root)}
    if extra_env:
        image_env.update(extra_env)
    return (
        modal.Image.debian_slim(python_version=VANILLA_SAM_PYTHON)
        .apt_install("git", "ca-certificates", "curl")
        .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
        .env({"PATH": MODAL_UV_PATH})
        .run_commands(
            f"uv pip install --system --python {VANILLA_SAM_PYTHON} "
            "numpy==1.26.4 pillow==11.0.0 opencv-python-headless==4.10.0.84 "
            "fastapi==0.118.0 pydantic==2.11.4 tqdm==4.67.1",
            f"uv pip install --system --python {VANILLA_SAM_PYTHON} "
            f"torch==2.5.1 torchvision==0.20.1 --index-url {PYTORCH_CU124_INDEX}",
            f"uv pip install --system --python {VANILLA_SAM_PYTHON} "
            '"segment-anything @ git+https://github.com/facebookresearch/segment-anything.git"',
        )
        .env(image_env)
        .add_local_dir(
            str(src),
            remote_path=str(remote_root / "src"),
            ignore=["**/__pycache__/**", "**/.DS_Store"],
        )
    )


def build_c3g_modal_image(
    *,
    cuda_image: str,
    torch_cuda_arch_list: str,
    repo_root: Path | None = None,
    workspace: Path = C3G_MODAL_WORKSPACE,
):
    """CUDA image for C3G on Modal: copy repo to ``/workspace`` and ``uv pip install -e``."""
    import modal

    if repo_root is None:
        repo_root = find_modal_repo_root()
        if repo_root is None:
            raise RuntimeError(
                "Could not find repo root (need pyproject.toml and src/). "
                "Run from the C3G checkout or pass repo_root= explicitly."
            )

    python_version = C3G_MODAL_PYTHON
    return (
        modal.Image.from_registry(cuda_image, add_python=python_version)
        .apt_install(
            "git",
            "curl",
            "ca-certificates",
            "build-essential",
            "clang",
            "libgl1",
            "libglib2.0-0",
        )
        .run_commands(
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            "echo 'export PATH=\"/root/.local/bin:$PATH\"' >> /root/.bashrc",
        )
        .env(
            {
                "PATH": MODAL_UV_PATH,
                "TORCH_CUDA_ARCH_LIST": torch_cuda_arch_list,
                "FORCE_CUDA": "1",
                "UV_INDEX_PYTORCH_CU124_URL": PYTORCH_CU124_INDEX,
            }
        )
        .add_local_dir(
            str(repo_root),
            remote_path=str(workspace),
            copy=True,
            ignore=[
                "**/.git/**",
                "**/__pycache__/**",
                "**/.venv/**",
                "**/datasets/**",
                "**/outputs/**",
                "**/.DS_Store",
                "src/dataset/replica_data/replica_semseg/**",
            ],
        )
        .workdir(str(workspace))
        .run_commands(
            f"cd {workspace} && uv pip install --system --python {python_version} -e ."
        )
        .env({"PYTHONPATH": str(workspace)})
    )


__all__ = [
    "C3G_MODAL_WORKSPACE",
    "C3G_SAM_EMA_EVAL_OUTPUT_MOUNT",
    "C3G_SAM_EMA_EVAL_OUTPUT_VOLUME",
    "C3G_SAM_EMA_MAG_UPROJ_EVAL_OUTPUT_MOUNT",
    "C3G_SAM_EMA_MAG_UPROJ_EVAL_OUTPUT_VOLUME",
    "C3G_SAM_EVAL_OUTPUT_MOUNT",
    "C3G_SAM_EVAL_OUTPUT_VOLUME",
    "C3G_SAM_NOEMA_NOMAG_EVAL_OUTPUT_MOUNT",
    "C3G_SAM_NOEMA_NOMAG_EVAL_OUTPUT_VOLUME",
    "DATASET_SPECS",
    "EXPERIMENT_PRED_ROOTS",
    "ExperimentName",
    "OUTPUT_MOUNT",
    "OUTPUT_VOLUME",
    "PRECOMPUTE_SAM_FEATURES_MOUNT",
    "PRECOMPUTE_SAM_FEATURES_VOLUME",
    "REPLICA_MOUNT",
    "REPLICA_VOLUME",
    "SCANNET_MOUNT",
    "SCANNET_VOLUME",
    "VANILLA_SAM_OUTPUT_MOUNT",
    "VANILLA_SAM_OUTPUT_VOLUME",
    "WEIGHTS_MOUNT",
    "WEIGHTS_VOLUME",
    "build_c3g_modal_image",
    "build_eval_sam_modal_image",
    "find_modal_repo_root",
    "resolve_dataset_root",
    "resolve_detach",
    "resolve_distillation_checkpoint",
    "REPLICA_2DSEG_SCENES",
    "SCANNET_2DSEG_SCENES",
    "SCANNET_2DSEG_TEST_SCENES",
    "expected_mask_class_ids",
    "find_smoke_frame",
    "find_smoke_scene",
    "iter_dataset_frames",
    "sample_eval_visualization_keys",
    "resolve_experiment",
    "resolve_sam_checkpoint",
    "run_subprocess_with_precompute_commit",
]
