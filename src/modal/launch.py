"""Dispatch C3G-SAM jobs to Modal from ``python -m src.main --modal …``."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

EVAL_ENTRYPOINTS = {
    "sam": "sam",
    "sam_smoke": "sam_smoke",
    "c3gsam": "c3g",
    "c3gsam_smoke": "c3g_smoke",
    "c3gsam_ema-mag-uproj": "c3gsam_ema-mag-uproj",
    "c3gsam_ema": "c3gsam_ema",
    "c3gsam_noema-nomag": "c3gsam_noema-nomag",
}


def _modal_cmd(script: str, entrypoint: str | None = None) -> list[str]:
    path = REPO_ROOT / script
    target = f"{path}::{entrypoint}" if entrypoint else str(path)
    return ["modal", "run", target]


def _run(cmd: list[str], *, wait: bool) -> None:
    if wait and "--wait" not in cmd:
        cmd = [*cmd, "--wait"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def run_modal_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run C3G-SAM workflows on Modal (use before Hydra local training)."
    )
    sub = parser.add_subparsers(dest="task", required=True)

    train_p = sub.add_parser("train", help="Train distillation or prompted SAM on Modal.")
    train_p.add_argument(
        "--experiment",
        choices=("distillation", "prompted"),
        default="distillation",
    )
    train_p.add_argument("--smoke", action="store_true")
    train_p.add_argument("--wait", action="store_true")

    pre_p = sub.add_parser("precompute", help="Precompute SAM encoder features on Modal.")
    pre_p.add_argument("--dataset", choices=("replica", "scannet"), default="scannet")
    pre_p.add_argument("--smoke", action="store_true")
    pre_p.add_argument("--wait", action="store_true")

    eval_p = sub.add_parser("eval", help="Export segmentation masks on Modal.")
    eval_p.add_argument(
        "--experiment",
        choices=tuple(EVAL_ENTRYPOINTS),
        default="c3gsam",
    )
    eval_p.add_argument("--dataset", choices=("replica", "scannet"), default="replica")
    eval_p.add_argument("--wait", action="store_true")

    score_p = sub.add_parser("score", help="Score exported masks on Modal.")
    score_p.add_argument(
        "--experiment",
        default="c3gsam",
        help="sam, c3gsam, c3gsam_ema-mag-uproj, c3gsam_ema, c3gsam_noema-nomag",
    )
    score_p.add_argument("--smoke", action="store_true")
    score_p.add_argument("--dataset", choices=("replica", "scannet"), default="replica")
    score_p.add_argument("--wait", action="store_true")

    viz_p = sub.add_parser("viz", help="Render segmentation comparison figures on Modal.")
    viz_p.add_argument("--wait", action="store_true")
    viz_p.add_argument("--output-dir", default=None)

    args = parser.parse_args(argv)

    if args.task == "train":
        entry = "smoke" if args.smoke else "main"
        cmd = _modal_cmd("src/modal/train.py", entry)
        if args.experiment != "distillation" or args.smoke:
            cmd.extend(["--experiment", args.experiment])
        _run(cmd, wait=args.wait)
        return

    if args.task == "precompute":
        entry = "smoke" if args.smoke else "main"
        cmd = _modal_cmd("src/modal/precompute.py", entry)
        cmd.extend(["--dataset", args.dataset])
        _run(cmd, wait=args.wait)
        return

    if args.task == "eval":
        entry = EVAL_ENTRYPOINTS[args.experiment]
        cmd = _modal_cmd("src/modal/eval_masks.py", entry)
        if entry == "sam_smoke":
            cmd.extend(["--dataset", args.dataset])
        _run(cmd, wait=args.wait)
        return

    if args.task == "score":
        entry = "smoke" if args.smoke else "main"
        cmd = _modal_cmd("src/modal/get_scores.py", entry)
        cmd.extend(["--experiment", args.experiment])
        if args.smoke:
            cmd.extend(["--dataset", args.dataset])
        _run(cmd, wait=args.wait)
        return

    if args.task == "viz":
        cmd = _modal_cmd("src/tools/seg_viz.py", "main")
        if args.output_dir:
            cmd.extend(["--output-dir", args.output_dir])
        _run(cmd, wait=args.wait)
        return

    parser.error(f"Unknown task {args.task!r}")


if __name__ == "__main__":
    run_modal_cli(sys.argv[1:])
