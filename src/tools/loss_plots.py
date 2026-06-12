"""Generate training loss figures from W&B-exported CSV logs.

Form1 plots use prompted ScanNet training (``feature_head_sam_prompted_scannet``).
W&B CSV columns ``loss:rgb_mse`` / ``loss:rgb_lpips`` hold **already-weighted**
``loss/mse`` and ``loss/lpips`` scalars from ``ModelWrapper.training_step``.
Coefficients come from ``config/loss/mse.yaml`` and ``config/loss/lpips.yaml``.

Form2 plots use SAM distillation training (``sam_distill_scannet``) from
``c3gsam:loss_*.csv`` exports. Validation plots use ``val_loss:feat.csv`` and
``val_loss:featmag.csv``; total validation distillation loss is computed from
the coefficients in ``config/training/feature_head_sam_precomputed.yaml``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

STEP_COLUMN = "trainer/global_step"
LINE_ALPHA = 0.7
TITLE_FONTSIZE = plt.rcParams["axes.titlesize"]
TICK_FONTSIZE = plt.rcParams["xtick.labelsize"]

# Weight applied inside LossMse / LossLpips before logging (see config/loss/*.yaml).
RGB_MSE_COEF = 1.0
RGB_LPIPS_COEF = 0.05
RGB_LOSS_TITLE = f"RGB Loss ({RGB_MSE_COEF}×MSE + {RGB_LPIPS_COEF}×LPIPS)"

# From config/training/feature_head_sam_precomputed.yaml (sam_distill_scannet).
FEATURE_COSINE_COEF = 1.0
FEATURE_MAG_COEF = 0.5
VAL_STEP_INTERVAL = 100


def _load_loss_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    value_columns = [
        col
        for col in df.columns
        if col != STEP_COLUMN
        and not col.endswith("__MIN")
        and not col.endswith("__MAX")
        and " - _step" not in col
    ]
    if not value_columns:
        raise ValueError(f"No loss value column found in {path}")
    return df[[STEP_COLUMN, value_columns[0]]].rename(
        columns={value_columns[0]: "value"}
    )


def _sample_every_n_steps(df: pd.DataFrame, interval: int) -> pd.DataFrame:
    return df.loc[df[STEP_COLUMN] % interval == 0].reset_index(drop=True)


def _plot_loss(
    ax: plt.Axes,
    steps: pd.Series,
    values: pd.Series,
    *,
    title: str,
    color: str,
) -> None:
    ax.plot(steps, values, linewidth=1.5, color=color, alpha=LINE_ALPHA)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)


def _configure_form2_step_axis(ax: plt.Axes, steps: pd.Series) -> None:
    step_min = float(steps.min())
    step_max = float(steps.max())
    ax.set_xlim(step_min, step_max)
    ticks = MaxNLocator(nbins=6, integer=True).tick_values(step_min, step_max)
    ticks = [tick for tick in ticks if tick != 5000]
    ax.set_xticks(ticks)


def plot_loss_figure(data_dir: Path, output_path: Path) -> None:
    total = _load_loss_csv(data_dir / "loss:total.csv")
    rgb_lpips = _load_loss_csv(data_dir / "loss:rgb_lpips.csv")
    rgb_mse = _load_loss_csv(data_dir / "loss:rgb_mse.csv")
    feature = _load_loss_csv(data_dir / "loss:feature.csv")
    seg = _load_loss_csv(data_dir / "loss:seg.csv")

    rgb = rgb_lpips.merge(rgb_mse, on=STEP_COLUMN, suffixes=("_lpips", "_mse"))
    rgb["value"] = rgb["value_lpips"] + rgb["value_mse"]

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], hspace=0.35, wspace=0.25)

    ax_total = fig.add_subplot(gs[0, :])
    ax_rgb = fig.add_subplot(gs[1, 0])
    ax_feature = fig.add_subplot(gs[1, 1], sharex=ax_rgb)
    ax_seg = fig.add_subplot(gs[1, 2], sharex=ax_rgb)

    _plot_loss(ax_total, total[STEP_COLUMN], total["value"], title="Total Loss", color="brown")
    _plot_loss(ax_rgb, rgb[STEP_COLUMN], rgb["value"], title=RGB_LOSS_TITLE, color="red")
    _plot_loss(
        ax_feature, feature[STEP_COLUMN], feature["value"], title="Feature Loss", color="green"
    )
    _plot_loss(
        ax_seg, seg[STEP_COLUMN], seg["value"], title="Segmentation Loss", color="blue"
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _combine_distill_total(
    feature: pd.DataFrame,
    feature_mag: pd.DataFrame,
) -> pd.DataFrame:
    merged = feature.merge(feature_mag, on=STEP_COLUMN, suffixes=("_cosine", "_mag"))
    merged["value"] = (
        FEATURE_COSINE_COEF * merged["value_cosine"]
        + FEATURE_MAG_COEF * merged["value_mag"]
    )
    return merged[[STEP_COLUMN, "value"]]


def plot_form2_total_loss_figure(data_dir: Path, output_path: Path) -> None:
    total = _load_loss_csv(data_dir / "c3gsam:loss_total.csv")

    fig, ax = plt.subplots(figsize=(10, 4))
    _plot_loss(ax, total[STEP_COLUMN], total["value"], title="Distillation Loss", color="brown")
    _configure_form2_step_axis(ax, total[STEP_COLUMN])

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_form2_feature_losses_figure(data_dir: Path, output_path: Path) -> None:
    feature = _load_loss_csv(data_dir / "c3gsam:loss_feat.csv")
    feature_mag = _load_loss_csv(data_dir / "c3gsam:loss_featmag.csv")

    fig, (ax_feature, ax_feature_mag) = plt.subplots(
        1, 2, figsize=(12, 4), sharex=True, gridspec_kw={"wspace": 0.25}
    )
    _plot_loss(
        ax_feature,
        feature[STEP_COLUMN],
        feature["value"],
        title="Feature Loss",
        color="green",
    )
    _plot_loss(
        ax_feature_mag,
        feature_mag[STEP_COLUMN],
        feature_mag["value"],
        title="Feature Magnitude Loss",
        color="orange",
    )
    _configure_form2_step_axis(ax_feature, feature[STEP_COLUMN])

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_form2_val_total_loss_figure(data_dir: Path, output_path: Path) -> None:
    feature = _sample_every_n_steps(
        _load_loss_csv(data_dir / "val_loss:feat.csv"), VAL_STEP_INTERVAL
    )
    feature_mag = _sample_every_n_steps(
        _load_loss_csv(data_dir / "val_loss:featmag.csv"), VAL_STEP_INTERVAL
    )
    total = _combine_distill_total(feature, feature_mag)

    fig, ax = plt.subplots(figsize=(10, 4))
    _plot_loss(ax, total[STEP_COLUMN], total["value"], title="Distillation Loss", color="brown")
    _configure_form2_step_axis(ax, total[STEP_COLUMN])

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_form2_val_feature_losses_figure(data_dir: Path, output_path: Path) -> None:
    feature = _sample_every_n_steps(
        _load_loss_csv(data_dir / "val_loss:feat.csv"), VAL_STEP_INTERVAL
    )
    feature_mag = _sample_every_n_steps(
        _load_loss_csv(data_dir / "val_loss:featmag.csv"), VAL_STEP_INTERVAL
    )

    fig, (ax_feature, ax_feature_mag) = plt.subplots(
        1, 2, figsize=(12, 4), sharex=True, gridspec_kw={"wspace": 0.25}
    )
    _plot_loss(
        ax_feature,
        feature[STEP_COLUMN],
        feature["value"],
        title="Feature Loss",
        color="green",
    )
    _plot_loss(
        ax_feature_mag,
        feature_mag[STEP_COLUMN],
        feature_mag["value"],
        title="Feature Magnitude Loss",
        color="orange",
    )
    _configure_form2_step_axis(ax_feature, feature[STEP_COLUMN])

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_form1_figures(data_dir: Path | None = None) -> list[Path]:
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / "c3gsam_results" / "form1"

    data_dir = data_dir.resolve()
    output_path = data_dir / "loss.png"
    plot_loss_figure(data_dir, output_path)
    return [output_path]


def generate_form2_figures(data_dir: Path | None = None) -> list[Path]:
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / "c3gsam_results" / "form2"

    data_dir = data_dir.resolve()
    total_path = data_dir / "dist_total.png"
    components_path = data_dir / "dist_components.png"
    val_total_path = data_dir / "dist_val_total.png"
    val_components_path = data_dir / "dist_val_components.png"
    plot_form2_total_loss_figure(data_dir, total_path)
    plot_form2_feature_losses_figure(data_dir, components_path)
    plot_form2_val_total_loss_figure(data_dir, val_total_path)
    plot_form2_val_feature_losses_figure(data_dir, val_components_path)
    return [total_path, components_path, val_total_path, val_components_path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training loss figures.")
    parser.add_argument(
        "--form",
        choices=("form1", "form2", "all"),
        default="all",
        help="Which result set to plot (default: all).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override data directory for the selected form.",
    )
    args = parser.parse_args()

    output_paths: list[Path] = []
    if args.form in ("form1", "all"):
        output_paths.extend(generate_form1_figures(args.data_dir if args.form == "form1" else None))
    if args.form in ("form2", "all"):
        output_paths.extend(generate_form2_figures(args.data_dir if args.form == "form2" else None))

    for output_path in output_paths:
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
