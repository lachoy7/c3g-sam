#!/usr/bin/env bash
# Modal C3G-SAM workflows (training, precompute, eval, scoring, viz).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WAIT=()
if [[ "${C3G_MODAL_WAIT:-0}" == "1" ]]; then
  WAIT=(--wait)
fi

usage() {
  cat <<'EOF'
Usage: scripts/run_modal.sh <command> [options]

Commands:
  precompute [--dataset replica|scannet] [--smoke] [--wait]
  train-distill [--smoke] [--wait]
  train-prompted [--smoke] [--wait]
  eval-sam [--smoke] [--wait]
  eval-c3gsam [--wait]
  eval-ablation <c3gsam_ema-mag-uproj|c3gsam_ema|c3gsam_noema-nomag> [--wait]
  score [--experiment NAME] [--smoke] [--wait]
  viz [--wait] [--output-dir PATH]
  data-examples [--wait]

All commands accept --wait to block until the Modal job finishes.
Set C3G_MODAL_WAIT=1 to wait by default.

Examples:
  scripts/run_modal.sh precompute --dataset scannet --wait
  scripts/run_modal.sh train-distill --wait
  scripts/run_modal.sh train-prompted --wait
  scripts/run_modal.sh eval-c3gsam --wait
  scripts/run_modal.sh eval-ablation c3gsam_ema-mag-uproj --wait
  scripts/run_modal.sh score --experiment c3gsam --wait
  scripts/run_modal.sh viz --wait

Equivalent via main.py:
  python -m src.main --modal train --experiment distillation --wait
  python -m src.main --modal precompute --dataset scannet --wait
  python -m src.main --modal eval --experiment c3gsam --wait
  python -m src.main --modal score --experiment c3gsam --wait
EOF
}

maybe_wait() {
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--wait" ]]; then
      WAIT=(--wait)
      shift
    else
      break
    fi
  done
  echo "$@"
}

cmd="${1:-}"
shift || true
maybe_wait "$@"
rest=("${WAIT[@]}" "$@")

case "$cmd" in
  precompute)
    dataset="scannet"
    entry="main"
    args=()
    for arg in "${rest[@]}"; do
      case "$arg" in
        --dataset) continue ;;
        --smoke) entry="smoke" ;;
        replica|scannet) dataset="$arg" ;;
        --wait) ;;
        *) args+=("$arg") ;;
      esac
    done
    i=0
    while [[ $i -lt ${#rest[@]} ]]; do
      if [[ "${rest[$i]}" == "--dataset" ]]; then
        dataset="${rest[$((i+1))]}"
        i=$((i+2))
      else
        i=$((i+1))
      fi
    done
    modal run src/modal/precompute.py::"$entry" --dataset "$dataset" "${WAIT[@]}"
    ;;
  train-distill)
    entry="main"
    args=(--experiment distillation)
    for arg in "${rest[@]}"; do
      [[ "$arg" == "--smoke" ]] && entry="smoke"
    done
    modal run src/modal/train.py::"$entry" "${args[@]}" "${WAIT[@]}"
    ;;
  train-prompted)
    entry="main"
    for arg in "${rest[@]}"; do
      [[ "$arg" == "--smoke" ]] && entry="smoke"
    done
    modal run src/modal/train.py::"$entry" --experiment prompted "${WAIT[@]}"
    ;;
  eval-sam)
    entry="sam"
    dataset_args=()
    for arg in "${rest[@]}"; do
      [[ "$arg" == "--smoke" ]] && entry="sam_smoke"
      [[ "$arg" == --dataset* ]] && dataset_args+=("$arg")
    done
    modal run src/modal/eval_masks.py::"$entry" "${dataset_args[@]}" "${WAIT[@]}"
    ;;
  eval-c3gsam)
    modal run src/modal/eval_masks.py::c3g "${WAIT[@]}"
    ;;
  eval-ablation)
    name="${1:-}"
    shift || true
    [[ -z "$name" ]] && { echo "Usage: run_modal.sh eval-ablation <experiment>"; exit 2; }
    modal run "src/modal/eval_masks.py::${name}" "${WAIT[@]}"
    ;;
  score)
    experiment="c3gsam"
    entry="main"
    extra=()
    i=0
    while [[ $i -lt ${#rest[@]} ]]; do
      arg="${rest[$i]}"
      case "$arg" in
        --experiment)
          experiment="${rest[$((i+1))]}"
          i=$((i+2))
          ;;
        --smoke)
          entry="smoke"
          i=$((i+1))
          ;;
        --wait)
          i=$((i+1))
          ;;
        *)
          extra+=("$arg")
          i=$((i+1))
          ;;
      esac
    done
    modal run src/modal/get_scores.py::"$entry" --experiment "$experiment" "${extra[@]}" "${WAIT[@]}"
    ;;
  viz)
    output_args=()
    for arg in "${rest[@]}"; do
      [[ "$arg" == --output-dir* ]] && output_args+=("$arg")
    done
    modal run src/tools/seg_viz.py "${output_args[@]}" "${WAIT[@]}"
    ;;
  data-examples)
    modal run src/tools/data_examples.py "${WAIT[@]}"
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
