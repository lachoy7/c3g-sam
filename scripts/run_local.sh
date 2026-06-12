#!/usr/bin/env bash
# Local C3G-SAM workflows (training, precompute, eval, scoring, viz).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: scripts/run_local.sh <command> [options]

Commands:
  precompute [--dataset replica|scannet] [--smoke]
  train-distill [--smoke] [extra hydra overrides...]
  train-prompted [--smoke] [extra hydra overrides...]
  eval-c3gsam [--checkpoint PATH] [hydra overrides...]
  eval-sam [--output-dir PATH]
  score [--experiment NAME] [--smoke] [--dataset replica|scannet]
  viz [--table-only]
  loss-plots [--form form1|form2|all]
  data-examples

Environment overrides:
  C3G_REPLICA_ROOT, C3G_SCANNET_ROOT, C3G_PRED_<EXPERIMENT>_ROOT

Examples:
  scripts/run_local.sh precompute --dataset scannet
  scripts/run_local.sh train-distill wandb.mode=disabled
  scripts/run_local.sh train-prompted --smoke
  scripts/run_local.sh eval-c3gsam checkpointing.load=./pretrained_weights/distillation-base.ckpt
  scripts/run_local.sh score --experiment c3gsam
  scripts/run_local.sh viz
EOF
}

run_python() {
  if command -v uv >/dev/null 2>&1; then
    uv run python "$@"
  else
    python3 "$@"
  fi
}

cmd="${1:-}"
shift || true

case "$cmd" in
  precompute)
    dataset="scannet"
    smoke=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --dataset) dataset="$2"; shift 2 ;;
        --smoke) smoke="--scenes $(ls datasets/$dataset 2>/dev/null | head -1)"; shift ;;
        *) echo "Unknown option: $1"; usage; exit 2 ;;
      esac
    done
    run_python scripts/precompute_sam_features.py --dataset "$dataset" --dataset-root "datasets/$dataset" --output-root "datasets/sam_features/$dataset" $smoke
    ;;
  train-distill)
    config="feature_head_sam_precomputed"
    if [[ "${1:-}" == "--smoke" ]]; then
      config="feature_head_sam_precomputed_smoke"
      shift
    fi
    run_python -m src.main +training="$config" "$@"
    ;;
  train-prompted)
    config="feature_head_sam_prompted_scannet"
    if [[ "${1:-}" == "--smoke" ]]; then
      config="feature_head_sam_prompted_scannet_smoke"
      shift
    fi
    run_python -m src.main +training="$config" "$@"
    ;;
  eval-c3gsam)
    run_python -m src.evaluation.mask_export +evaluation=c3g_sam_distill "$@"
    ;;
  eval-sam)
    output="outputs/vanilla-sam"
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --output-dir) output="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; usage; exit 2 ;;
      esac
    done
    run_python - <<PY
from pathlib import Path
import torch
from src.evaluation.mask_export import export_vanilla_sam_masks
from src.model.sam import load_sam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = load_sam("sam_vit_h", "./pretrained_weights/sam_vit_h.pth", freeze=True).to(device)
export_vanilla_sam_masks(
    sam,
    device,
    output_root=Path("$output"),
    dataset_roots={
        "replica": Path("datasets/replica"),
        "scannet": Path("datasets/scannet"),
    },
)
PY
    ;;
  score)
    experiment="c3gsam"
    extra=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --experiment) experiment="$2"; shift 2 ;;
        --smoke) extra+=(--smoke); shift ;;
        --dataset) extra+=(--dataset "$2"); shift 2 ;;
        *) extra+=("$1"); shift ;;
      esac
    done
    run_python -m src.evaluation.score_masks --experiment "$experiment" "${extra[@]}"
    ;;
  viz)
    run_python -m src.tools.seg_viz "$@"
    ;;
  loss-plots)
    run_python -m src.tools.loss_plots "$@"
    ;;
  data-examples)
    run_python -m src.tools.data_examples "$@"
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "Unknown command: $cmd"
    usage
    exit 2
    ;;
esac
