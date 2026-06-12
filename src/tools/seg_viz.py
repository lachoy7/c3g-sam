#!/usr/bin/env python3
"""Visualize dense segmentation masks from Modal eval volumes.

For each eval output volume (vanilla SAM + C3G-SAM variants), picks two consecutive
frames from one random test scene per dataset (Replica + ScanNet), loads
``dense_mask.png`` predictions (or merges per-class exports), and writes a
side-by-side colored figure. Ground-truth label maps get the same treatment with
identical class colors across all figures (background = black).

Examples::

    modal run src/tools/seg_viz.py --wait
    modal run src/tools/seg_viz.py --output-dir c3gsam_results/seg_results --wait
    python -m src.tools.seg_viz \\
        --replica-root /path/to/replica \\
        --scannet-root /path/to/scannet \\
        --sam-root /path/to/vanilla-sam-outputs \\
        --c3gsam-root /path/to/c3g-sam-eval-outputs \\
        --output-dir c3gsam_results/seg_results
    python -m src.tools.seg_viz --table-only
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import modal

from src.evaluation.eval_common import (
    LOCAL_EXPERIMENT_PRED_ROOTS,
    resolve_local_dataset_root,
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
    REPLICA_2DSEG_SCENES,
    REPLICA_MOUNT,
    REPLICA_VOLUME,
    SCANNET_2DSEG_TEST_SCENES,
    SCANNET_MOUNT,
    SCANNET_VOLUME,
    VANILLA_SAM_OUTPUT_MOUNT,
    VANILLA_SAM_OUTPUT_VOLUME,
    build_eval_sam_modal_image,
    expected_mask_class_ids,
    resolve_dataset_root,
    resolve_detach,
)

APP_NAME = "c3g-seg-viz"
DEFAULT_SEED = 42
DEFAULT_MIN_OBJECT_PIXELS = 16
DENSE_MASK_FILENAME = "dense_mask.png"
REMOTE_SCRATCH_DIR = Path("/tmp/seg_viz")


def default_local_output_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "c3gsam_results" / "seg_results"

DatasetName = Literal["replica", "scannet"]

TEST_SCENES: dict[DatasetName, list[str]] = {
    "replica": list(REPLICA_2DSEG_SCENES),
    "scannet": list(SCANNET_2DSEG_TEST_SCENES),
}

TableExperiment = ExperimentName | Literal["gt"]
ComparisonTableRow = tuple[TableExperiment, str]

COMPARISON_TABLE_ROWS: tuple[ComparisonTableRow, ...] = (
    ("gt", "GT"),
    ("sam", "SAM"),
    ("c3gsam", "C3G-SAM"),
    ("c3gsam_ema-mag-uproj", "C3G-SAM (EMA + mag + up-proj)"),
    ("c3gsam_ema", "C3G-SAM (no mag head)"),
    ("c3gsam_noema-nomag", "C3G-SAM (EMA, no mag)"),
)
COMPARISON_TABLE_FILENAME = "comparison_table.png"
COMPARISON_TABLE_MAIN_FILENAME = "comparison_table_main.png"
COMPARISON_TABLE_ABLATION_FILENAME = "comparison_table_ablation.png"
COMPARISON_TABLE_CONFIGS: tuple[tuple[str, tuple[ComparisonTableRow, ...]], ...] = (
    (
        COMPARISON_TABLE_MAIN_FILENAME,
        (
            ("gt", "GT"),
            ("sam", "SAM"),
            ("c3gsam", "C3G-SAM"),
        ),
    ),
    (
        COMPARISON_TABLE_ABLATION_FILENAME,
        (
            ("gt", "GT"),
            ("c3gsam_ema", "no EMA + no mag head"),
            ("c3gsam_noema-nomag", "EMA + no mag head"),
            ("c3gsam", "EMA + mag head + zero pad"),
            ("c3gsam_ema-mag-uproj", "EMA + mag head + up proj"),
        ),
    ),
)
MASK_FIGURE_HEADER_HEIGHT = 48
TABLE_FONT_SIZE = 40
TABLE_HEADER_FONT_SIZE = 44
TABLE_LABEL_PADDING = 28
TABLE_HEADER_PADDING = 20

# Same palette as src/visualization/colors.py, excluding black/white for foreground.
FOREGROUND_HEX_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#808080",
]

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

viz_image = build_eval_sam_modal_image()

SEG_VIZ_VOLUMES = {
    str(REPLICA_MOUNT): replica_volume,
    str(SCANNET_MOUNT): scannet_volume,
    str(VANILLA_SAM_OUTPUT_MOUNT): vanilla_output_volume,
    str(C3G_SAM_EVAL_OUTPUT_MOUNT): c3g_eval_output_volume,
    str(C3G_SAM_EMA_MAG_UPROJ_EVAL_OUTPUT_MOUNT): c3g_ema_mag_uproj_eval_output_volume,
    str(C3G_SAM_EMA_EVAL_OUTPUT_MOUNT): c3g_ema_eval_output_volume,
    str(C3G_SAM_NOEMA_NOMAG_EVAL_OUTPUT_MOUNT): c3g_noema_nomag_eval_output_volume,
}


@dataclass(frozen=True, slots=True)
class FramePairSelection:
    dataset: DatasetName
    scene_id: str
    frame_a: str
    frame_b: str


def _load_pred_mask(pred_path: Path):
    import numpy as np
    from PIL import Image

    if not pred_path.is_file():
        return None
    return np.array(Image.open(pred_path))


def _pred_mask_to_bool(mask):
    import numpy as np

    if mask.dtype == np.bool_:
        return mask
    return mask > 128


def _resize_logits_to_label_shape(logits, label_shape: tuple[int, int]):
    import cv2

    if tuple(logits.shape[:2]) == label_shape:
        return logits
    height, width = label_shape
    return cv2.resize(
        logits.astype("float32", copy=False),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def _build_dense_pred_mask(
    frame_pred_dir: Path,
    label_shape: tuple[int, int],
    class_ids: list[int],
) -> tuple:
    import numpy as np

    dense = np.zeros(label_shape, dtype=np.int32)
    best_logit = np.full(label_shape, -np.inf, dtype=np.float32)
    missing_predictions = 0
    for class_id in class_ids:
        binary = _load_pred_mask(frame_pred_dir / f"{class_id}.png")
        logits_path = frame_pred_dir / f"{class_id}_logits.npy"
        if binary is None or not logits_path.is_file():
            missing_predictions += 1
            continue
        mask = _pred_mask_to_bool(binary)
        logits = np.load(logits_path).astype(np.float32)
        if tuple(logits.shape[:2]) != label_shape:
            logits = _resize_logits_to_label_shape(logits, label_shape)
        update = mask & (logits > best_logit)
        dense[update] = class_id
        best_logit[update] = logits[update]
    return dense, missing_predictions


def _load_dense_mask(
    frame_pred_dir: Path,
    label_path: Path,
    *,
    min_object_pixels: int,
):
    import numpy as np
    from PIL import Image

    gt_dense = np.array(Image.open(label_path))
    dense_path = frame_pred_dir / DENSE_MASK_FILENAME
    if dense_path.is_file():
        dense = np.array(Image.open(dense_path))
        if dense.shape[:2] != gt_dense.shape[:2]:
            raise ValueError(
                f"Dense mask shape {dense.shape[:2]} != label shape "
                f"{gt_dense.shape[:2]} at {dense_path}"
            )
        return dense.astype(np.int32)

    class_ids = expected_mask_class_ids(
        label_path, min_object_pixels=min_object_pixels
    )
    dense, _ = _build_dense_pred_mask(
        frame_pred_dir, gt_dense.shape[:2], class_ids
    )
    return dense


def _load_gt_dense(label_path: Path):
    import numpy as np
    from PIL import Image

    return np.array(Image.open(label_path)).astype(np.int32)


def _frame_has_predictions(
    frame_pred_dir: Path,
    label_path: Path,
    *,
    min_object_pixels: int,
) -> bool:
    if (frame_pred_dir / DENSE_MASK_FILENAME).is_file():
        return True
    if not frame_pred_dir.is_dir():
        return False
    class_ids = expected_mask_class_ids(
        label_path, min_object_pixels=min_object_pixels
    )
    if not class_ids:
        return True
    return all(
        (frame_pred_dir / f"{class_id}.png").is_file()
        and (frame_pred_dir / f"{class_id}_logits.npy").is_file()
        for class_id in class_ids
    )


def _frame_pair_candidates(
    *,
    dataset: DatasetName,
    dataset_root: Path,
    pred_roots: dict[ExperimentName, Path],
    scenes: list[str],
    min_object_pixels: int,
) -> list[FramePairSelection]:
    from src.misc.frame_layout import FramePaths, list_frame_ids

    candidates: list[FramePairSelection] = []
    for scene_id in scenes:
        scene_dir = dataset_root / scene_id
        if not scene_dir.is_dir():
            continue

        frame_ids = list_frame_ids(scene_dir)
        for frame_index in range(len(frame_ids) - 1):
            frame_a = frame_ids[frame_index]
            frame_b = frame_ids[frame_index + 1]
            paths_a = FramePaths.from_frame_id(scene_dir, frame_a)
            paths_b = FramePaths.from_frame_id(scene_dir, frame_b)
            if not (paths_a.label.is_file() and paths_b.label.is_file()):
                continue

            complete = True
            for pred_root in pred_roots.values():
                pred_scene_dir = pred_root / dataset / scene_id
                for frame_id, label_path in (
                    (frame_a, paths_a.label),
                    (frame_b, paths_b.label),
                ):
                    if not _frame_has_predictions(
                        pred_scene_dir / frame_id,
                        label_path,
                        min_object_pixels=min_object_pixels,
                    ):
                        complete = False
                        break
                if not complete:
                    break

            if complete:
                candidates.append(
                    FramePairSelection(
                        dataset=dataset,
                        scene_id=scene_id,
                        frame_a=frame_a,
                        frame_b=frame_b,
                    )
                )
    return candidates


def _pick_frame_pairs(
    *,
    replica_root: Path,
    scannet_root: Path,
    pred_roots: dict[ExperimentName, Path],
    seed: int,
    min_object_pixels: int,
) -> dict[DatasetName, FramePairSelection]:
    rng = random.Random(seed)
    selections: dict[DatasetName, FramePairSelection] = {}

    for dataset, dataset_root in (
        ("replica", replica_root),
        ("scannet", scannet_root),
    ):
        candidates = _frame_pair_candidates(
            dataset=dataset,  # type: ignore[arg-type]
            dataset_root=dataset_root,
            pred_roots=pred_roots,
            scenes=TEST_SCENES[dataset],  # type: ignore[index]
            min_object_pixels=min_object_pixels,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No scene with consecutive labeled frames and complete predictions "
                f"for all experiments under {dataset_root}."
            )
        selections[dataset] = rng.choice(candidates)  # type: ignore[index]

    return selections


def _collect_class_ids(
    *,
    selections: dict[DatasetName, FramePairSelection],
    replica_root: Path,
    scannet_root: Path,
    pred_roots: dict[ExperimentName, Path],
    min_object_pixels: int,
) -> set[int]:
    import numpy as np

    from src.misc.frame_layout import FramePaths

    class_ids: set[int] = set()
    dataset_roots = {"replica": replica_root, "scannet": scannet_root}

    for dataset, selection in selections.items():
        scene_dir = dataset_roots[dataset] / selection.scene_id
        for frame_id in (selection.frame_a, selection.frame_b):
            label_path = FramePaths.from_frame_id(scene_dir, frame_id).label
            gt_dense = _load_gt_dense(label_path)
            class_ids.update(
                int(class_id) for class_id in np.unique(gt_dense) if class_id
            )
            for pred_root in pred_roots.values():
                pred_dense = _load_dense_mask(
                    pred_root / dataset / selection.scene_id / frame_id,
                    label_path,
                    min_object_pixels=min_object_pixels,
                )
                class_ids.update(
                    int(class_id)
                    for class_id in pred_dense.flat
                    if class_id != 0
                )

    class_ids.discard(0)
    return class_ids


def _build_class_colormap(class_ids: set[int]) -> dict[int, tuple[int, int, int]]:
    from PIL import ImageColor

    colormap: dict[int, tuple[int, int, int]] = {0: (0, 0, 0)}
    for index, class_id in enumerate(sorted(class_ids)):
        hex_color = FOREGROUND_HEX_COLORS[index % len(FOREGROUND_HEX_COLORS)]
        colormap[class_id] = ImageColor.getcolor(hex_color, "RGB")
    return colormap


def _colorize_dense_mask(dense, colormap: dict[int, tuple[int, int, int]]):
    import numpy as np

    rgb = np.zeros((*dense.shape[:2], 3), dtype=np.uint8)
    for class_id, color in colormap.items():
        rgb[dense == class_id] = color
    return rgb


def _save_two_mask_figure(
    *,
    mask_a,
    mask_b,
    colormap: dict[int, tuple[int, int, int]],
    title: str,
    subtitles: tuple[str, str],
    output_path: Path,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    rgb_a = _colorize_dense_mask(mask_a, colormap)
    rgb_b = _colorize_dense_mask(mask_b, colormap)
    gap = 8
    header_height = MASK_FIGURE_HEADER_HEIGHT
    panel_width = rgb_a.shape[1] + rgb_b.shape[1] + gap
    panel_height = max(rgb_a.shape[0], rgb_b.shape[0])
    canvas = Image.new(
        "RGB",
        (panel_width, header_height + panel_height),
        color=(0, 0, 0),
    )
    canvas.paste(Image.fromarray(rgb_a), (0, header_height))
    canvas.paste(Image.fromarray(rgb_b), (rgb_a.shape[1] + gap, header_height))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((4, 4), title, fill=(255, 255, 255), font=font)
    draw.text((4, 22), f"{subtitles[0]}  |  {subtitles[1]}", fill=(200, 200, 200), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _strip_figure_header(image, header_height: int = MASK_FIGURE_HEADER_HEIGHT):
    if image.height <= header_height:
        return image
    return image.crop((0, header_height, image.width, image.height))


def _load_table_font(size: int):
    from PIL import ImageFont

    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _label_column_width(labels: list[str], font, *, padding: int = 16) -> int:
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    widths = [
        draw.textbbox((0, 0), label, font=font)[2]
        - draw.textbbox((0, 0), label, font=font)[0]
        for label in labels
    ]
    return max(widths, default=0) + padding


def _crop_center_to_size(image, width: int, height: int):
    from PIL import Image

    if image.size == (width, height):
        return image
    scale = max(width / image.width, height / image.height)
    scaled = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = max(0, (scaled.width - width) // 2)
    top = max(0, (scaled.height - height) // 2)
    return scaled.crop((left, top, left + width, top + height))


def _table_cell_size(
    replica_image,
    scannet_image,
    *,
    max_cell_width: int,
) -> tuple[int, int]:
    reference = replica_image or scannet_image
    assert reference is not None
    reference = _strip_figure_header(reference)
    width, height = reference.size
    if width > max_cell_width:
        scale = max_cell_width / width
        width = max_cell_width
        height = max(1, round(height * scale))
    return width, height


def _prepare_table_cell(image, target_size: tuple[int, int]):
    stripped = _strip_figure_header(image)
    return _crop_center_to_size(stripped, target_size[0], target_size[1])


def _draw_centered_text(
    draw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    fill: tuple[int, int, int] = (255, 255, 255),
    font,
) -> None:
    left, top, right, bottom = box
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = left + max(0, (right - left - text_width) // 2)
    y = top + max(0, (bottom - top - text_height) // 2)
    draw.text((x, y), text, fill=fill, font=font)


def build_seg_comparison_table(
    *,
    input_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    rows: tuple[ComparisonTableRow, ...] = COMPARISON_TABLE_ROWS,
    max_cell_width: int = 640,
) -> Path | None:
    """Compose a method | ScanNet | Replica table from existing seg PNGs."""
    from PIL import Image, ImageDraw

    in_dir = (input_dir or default_local_output_dir()).resolve()
    out_path = (
        Path(output_path)
        if output_path is not None
        else in_dir / COMPARISON_TABLE_FILENAME
    ).resolve()

    font = _load_table_font(TABLE_FONT_SIZE)
    header_font = _load_table_font(TABLE_HEADER_FONT_SIZE)
    row_gap = 2
    column_gap = 2
    margin = 2
    placeholder_color = (24, 24, 24)

    loaded_rows: list[tuple[str, Image.Image | None, Image.Image | None]] = []
    for experiment, label in rows:
        scannet_path = in_dir / f"{experiment}_scannet.png"
        replica_path = in_dir / f"{experiment}_replica.png"
        scannet_image = (
            Image.open(scannet_path).convert("RGB")
            if scannet_path.is_file()
            else None
        )
        replica_image = (
            Image.open(replica_path).convert("RGB")
            if replica_path.is_file()
            else None
        )
        if scannet_image is None and replica_image is None:
            continue
        loaded_rows.append((label, scannet_image, replica_image))

    if not loaded_rows:
        print(f"No seg figures found under {in_dir} for {out_path.name}")
        return None
    labels = [label for label, _, _ in loaded_rows]
    label_column_width = _label_column_width(
        labels, font, padding=TABLE_LABEL_PADDING
    )
    header_height = TABLE_HEADER_FONT_SIZE + TABLE_HEADER_PADDING

    prepared_rows: list[tuple[str, Image.Image | None, Image.Image | None, tuple[int, int]]] = []
    for label, scannet_image, replica_image in loaded_rows:
        cell_size = _table_cell_size(
            replica_image,
            scannet_image,
            max_cell_width=max_cell_width,
        )
        prepared_rows.append(
            (
                label,
                _prepare_table_cell(scannet_image, cell_size)
                if scannet_image is not None
                else None,
                _prepare_table_cell(replica_image, cell_size)
                if replica_image is not None
                else None,
                cell_size,
            )
        )

    cell_width, cell_height = prepared_rows[0][3]
    body_height = cell_height * len(prepared_rows) + row_gap * (len(prepared_rows) - 1)
    canvas_width = (
        margin * 2
        + label_column_width
        + column_gap
        + cell_width
        + column_gap
        + cell_width
    )
    canvas_height = margin * 2 + header_height + row_gap + body_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    scannet_left = margin + label_column_width + column_gap
    replica_left = scannet_left + cell_width + column_gap
    header_top = margin
    _draw_centered_text(
        draw,
        (margin, header_top, margin + label_column_width, header_top + header_height),
        "Method",
        font=header_font,
    )
    _draw_centered_text(
        draw,
        (scannet_left, header_top, scannet_left + cell_width, header_top + header_height),
        "ScanNet",
        font=header_font,
    )
    _draw_centered_text(
        draw,
        (replica_left, header_top, replica_left + cell_width, header_top + header_height),
        "Replica",
        font=header_font,
    )

    y = margin + header_height + row_gap
    for label, scannet_image, replica_image, (row_cell_width, row_cell_height) in prepared_rows:
        _draw_centered_text(
            draw,
            (margin, y, margin + label_column_width, y + row_cell_height),
            label,
            font=font,
        )

        for column_left, image in (
            (scannet_left, scannet_image),
            (replica_left, replica_image),
        ):
            if image is None:
                placeholder = Image.new(
                    "RGB",
                    (row_cell_width, row_cell_height),
                    color=placeholder_color,
                )
                canvas.paste(placeholder, (column_left, y))
                continue
            canvas.paste(image, (column_left, y))

        y += row_cell_height + row_gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"Wrote comparison table to {out_path}")
    return out_path


def build_all_comparison_tables(
    *,
    input_dir: str | Path | None = None,
    max_cell_width: int = 640,
) -> list[Path]:
    """Build the preset main and ablation comparison tables."""
    in_dir = (input_dir or default_local_output_dir()).resolve()
    written: list[Path] = []
    for filename, rows in COMPARISON_TABLE_CONFIGS:
        path = build_seg_comparison_table(
            input_dir=in_dir,
            output_path=in_dir / filename,
            rows=rows,
            max_cell_width=max_cell_width,
        )
        if path is not None:
            written.append(path)
    return written


def _dataset_roots(replica_root: str, scannet_root: str) -> dict[DatasetName, Path]:
    return {
        "replica": Path(resolve_dataset_root("replica", replica_root)),
        "scannet": Path(resolve_dataset_root("scannet", scannet_root)),
    }


def _resolve_pred_roots(
    overrides: dict[ExperimentName, str | None] | None = None,
    *,
    default_roots: dict[ExperimentName, Path] | None = None,
) -> dict[ExperimentName, Path]:
    overrides = overrides or {}
    roots: dict[ExperimentName, Path] = {}
    base_roots = default_roots or EXPERIMENT_PRED_ROOTS
    for experiment, default_root in base_roots.items():
        override = overrides.get(experiment)
        roots[experiment] = Path(override) if override else default_root
    return roots


def generate_seg_viz_figures(
    *,
    replica_root: str | Path,
    scannet_root: str | Path,
    pred_roots: dict[ExperimentName, Path],
    output_dir: str | Path,
    seed: int = DEFAULT_SEED,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
) -> dict:
    from src.misc.frame_layout import FramePaths

    replica_path = Path(replica_root)
    scannet_path = Path(scannet_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selections = _pick_frame_pairs(
        replica_root=replica_path,
        scannet_root=scannet_path,
        pred_roots=pred_roots,
        seed=seed,
        min_object_pixels=min_object_pixels,
    )
    colormap = _build_class_colormap(
        _collect_class_ids(
            selections=selections,
            replica_root=replica_path,
            scannet_root=scannet_path,
            pred_roots=pred_roots,
            min_object_pixels=min_object_pixels,
        )
    )

    written: list[str] = []
    dataset_roots = {"replica": replica_path, "scannet": scannet_path}

    for dataset, selection in selections.items():
        scene_dir = dataset_roots[dataset] / selection.scene_id
        paths_a = FramePaths.from_frame_id(scene_dir, selection.frame_a)
        paths_b = FramePaths.from_frame_id(scene_dir, selection.frame_b)
        gt_a = _load_gt_dense(paths_a.label)
        gt_b = _load_gt_dense(paths_b.label)
        subtitles = (
            f"{selection.frame_a}",
            f"{selection.frame_b}",
        )
        gt_path = out_dir / f"gt_{dataset}.png"
        _save_two_mask_figure(
            mask_a=gt_a,
            mask_b=gt_b,
            colormap=colormap,
            title=(
                f"GT — {dataset} / {selection.scene_id} "
                f"({selection.frame_a}, {selection.frame_b})"
            ),
            subtitles=subtitles,
            output_path=gt_path,
        )
        written.append(str(gt_path))

        for experiment, pred_root in pred_roots.items():
            pred_scene_dir = pred_root / dataset / selection.scene_id
            pred_a = _load_dense_mask(
                pred_scene_dir / selection.frame_a,
                paths_a.label,
                min_object_pixels=min_object_pixels,
            )
            pred_b = _load_dense_mask(
                pred_scene_dir / selection.frame_b,
                paths_b.label,
                min_object_pixels=min_object_pixels,
            )
            pred_path = out_dir / f"{experiment}_{dataset}.png"
            _save_two_mask_figure(
                mask_a=pred_a,
                mask_b=pred_b,
                colormap=colormap,
                title=(
                    f"{experiment} — {dataset} / {selection.scene_id} "
                    f"({selection.frame_a}, {selection.frame_b})"
                ),
                subtitles=subtitles,
                output_path=pred_path,
            )
            written.append(str(pred_path))

    payload = {
        "output_dir": str(out_dir),
        "seed": seed,
        "colormap_class_ids": sorted(k for k in colormap if k != 0),
        "selections": {
            dataset: {
                "scene_id": selection.scene_id,
                "frame_a": selection.frame_a,
                "frame_b": selection.frame_b,
            }
            for dataset, selection in selections.items()
        },
        "figures": written,
    }
    print(json_dumps(payload))
    return payload


def json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2) + "\n"


def _attach_figure_bytes(report: dict) -> dict:
    report["figure_bytes"] = {
        Path(figure_path).name: Path(figure_path).read_bytes()
        for figure_path in report["figures"]
    }
    return report


def save_figures_locally(report: dict, output_dir: Path | None = None) -> Path:
    """Write PNGs returned from a Modal run into a local directory."""
    out_dir = (output_dir or default_local_output_dir()).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    local_paths: list[str] = []
    for name, data in report.get("figure_bytes", {}).items():
        path = out_dir / name
        path.write_bytes(data)
        local_paths.append(str(path))

    manifest = {key: value for key, value in report.items() if key != "figure_bytes"}
    manifest["figures"] = local_paths
    manifest["output_dir"] = str(out_dir)
    (out_dir / "manifest.json").write_text(json_dumps(manifest), encoding="utf-8")
    print(f"Wrote {len(local_paths)} figures to {out_dir}")
    return out_dir


@app.function(
    image=viz_image,
    cpu=4,
    memory=8192,
    timeout=60 * 30,
    volumes=SEG_VIZ_VOLUMES,
    nonpreemptible=True,
)
def render_seg_viz(
    replica_root: str | None = None,
    scannet_root: str | None = None,
    output_dir: str | None = None,
    seed: int = DEFAULT_SEED,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
) -> dict:
    """Render segmentation comparison figures on Modal worker volumes."""
    del output_dir  # Figures are always pulled to the local machine after the run.
    pred_roots = _resolve_pred_roots()
    out_dir = REMOTE_SCRATCH_DIR

    for dataset_name, root in _dataset_roots(
        resolve_dataset_root("replica", replica_root),
        resolve_dataset_root("scannet", scannet_root),
    ).items():
        if not root.is_dir():
            raise FileNotFoundError(
                f"{dataset_name} dataset not found at {root}. "
                f"Populate the `{dataset_name}` volume."
            )

    for experiment, pred_root in pred_roots.items():
        if not pred_root.is_dir():
            raise FileNotFoundError(
                f"Predictions for {experiment!r} not found at {pred_root}."
            )

    report = generate_seg_viz_figures(
        replica_root=resolve_dataset_root("replica", replica_root),
        scannet_root=resolve_dataset_root("scannet", scannet_root),
        pred_roots=pred_roots,
        output_dir=out_dir,
        seed=seed,
        min_object_pixels=min_object_pixels,
    )
    return _attach_figure_bytes(report)


def _dispatch(fn, *, job_name: str, detach: bool, **kwargs):
    from src.misc.modal_run import dispatch_remote

    return dispatch_remote(
        fn,
        detach=detach,
        job_name=job_name,
        app_name=APP_NAME,
        **kwargs,
    )


@app.local_entrypoint()
def main(
    replica_root: str | None = None,
    scannet_root: str | None = None,
    output_dir: str | None = None,
    seed: int = DEFAULT_SEED,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
    detach: bool | None = None,
    wait: bool = False,
) -> None:
    """Render seg viz figures from Modal eval volumes."""
    detached = resolve_detach(detach=detach, remote_job=not wait)
    report = _dispatch(
        render_seg_viz,
        job_name="segmentation viz",
        detach=detached,
        replica_root=replica_root,
        scannet_root=scannet_root,
        output_dir=output_dir,
        seed=seed,
        min_object_pixels=min_object_pixels,
    )
    if detached:
        print(
            "Detached run: figures are not saved locally. "
            "Re-run with --wait to write into c3gsam_results/seg_results/."
        )
        return

    local_out = Path(output_dir) if output_dir else default_local_output_dir()
    save_figures_locally(report, local_out)
    build_all_comparison_tables(input_dir=local_out)


def _parse_local_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render colored dense-mask figures from eval outputs."
    )
    parser.add_argument(
        "--table-only",
        action="store_true",
        help=(
            "Build preset comparison tables from existing PNG files in "
            "--output-dir (skip eval figure generation)."
        ),
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=None,
        help=(
            "Write a single full comparison table to this path instead of the "
            "preset main/ablation tables."
        ),
    )
    parser.add_argument("--replica-root", type=Path, default=None)
    parser.add_argument("--scannet-root", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_local_output_dir(),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--min-object-pixels",
        type=int,
        default=DEFAULT_MIN_OBJECT_PIXELS,
    )
    parser.add_argument("--sam-root", type=Path, default=None)
    parser.add_argument("--c3gsam-root", type=Path, default=None)
    parser.add_argument("--c3gsam-ema-mag-uproj-root", type=Path, default=None)
    parser.add_argument("--c3gsam-ema-root", type=Path, default=None)
    parser.add_argument("--c3gsam-noema-nomag-root", type=Path, default=None)
    return parser.parse_args(argv)


def main_local(argv: list[str] | None = None) -> None:
    args = _parse_local_args(argv)
    if args.table_only:
        if args.table_output is not None:
            build_seg_comparison_table(
                input_dir=args.output_dir,
                output_path=args.table_output,
            )
        else:
            build_all_comparison_tables(input_dir=args.output_dir)
        return

    pred_roots = _resolve_pred_roots(
        {
            "sam": args.sam_root,
            "c3gsam": args.c3gsam_root,
            "c3gsam_ema-mag-uproj": args.c3gsam_ema_mag_uproj_root,
            "c3gsam_ema": args.c3gsam_ema_root,
            "c3gsam_noema-nomag": args.c3gsam_noema_nomag_root,
        },
        default_roots=LOCAL_EXPERIMENT_PRED_ROOTS,
    )
    generate_seg_viz_figures(
        replica_root=args.replica_root or resolve_local_dataset_root("replica"),
        scannet_root=args.scannet_root or resolve_local_dataset_root("scannet"),
        pred_roots=pred_roots,
        output_dir=args.output_dir,
        seed=args.seed,
        min_object_pixels=args.min_object_pixels,
    )
    build_all_comparison_tables(input_dir=args.output_dir)


if __name__ == "__main__":
    if "modal" in sys.modules and hasattr(modal, "is_local") and not modal.is_local():
        raise SystemExit(
            "Run with `modal run src/tools/seg_viz.py`, not as a worker entrypoint."
        )
    main_local()
