#!/usr/bin/env python3
"""Render dataset input / GT-mask examples for Replica and ScanNet.

Builds a 2×4 figure: top row RGB images, bottom row colored GT masks, with two
examples per dataset (ScanNet columns 0–1, Replica columns 2–3). Styling
matches ``seg_viz.py``.

Examples::

    modal run src/tools/data_examples.py --wait
    python -m src.tools.data_examples \\
        --replica-root /path/to/replica \\
        --scannet-root /path/to/scannet
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import modal

from src.modal.common import (
    REPLICA_MOUNT,
    REPLICA_VOLUME,
    SCANNET_MOUNT,
    SCANNET_VOLUME,
    build_eval_sam_modal_image,
    resolve_dataset_root,
    resolve_detach,
)
from src.tools.seg_viz import (
    DEFAULT_SEED,
    DatasetName,
    TABLE_HEADER_FONT_SIZE,
    TABLE_HEADER_PADDING,
    TEST_SCENES,
    _build_class_colormap,
    _colorize_dense_mask,
    _crop_center_to_size,
    _draw_centered_text,
    _load_gt_dense,
    _load_table_font,
)

APP_NAME = "c3g-data-examples"
EXAMPLES_PER_DATASET = 2
NUM_COLUMNS = EXAMPLES_PER_DATASET * 2
NUM_ROWS = 2
REMOTE_SCRATCH_DIR = Path("/tmp/data_examples")

DATASET_COLUMNS: tuple[tuple[DatasetName, str], ...] = (
    ("scannet", "ScanNet"),
    ("replica", "Replica"),
)

app = modal.App(APP_NAME)
replica_volume = modal.Volume.from_name(REPLICA_VOLUME, create_if_missing=True)
scannet_volume = modal.Volume.from_name(SCANNET_VOLUME, create_if_missing=True)
viz_image = build_eval_sam_modal_image()

DATA_EXAMPLES_VOLUMES = {
    str(REPLICA_MOUNT): replica_volume,
    str(SCANNET_MOUNT): scannet_volume,
}


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "c3gsam_results"
        / "data_examples.png"
    )


def default_local_dataset_root(dataset: DatasetName) -> Path:
    return Path(__file__).resolve().parents[2] / "datasets" / dataset


@dataclass(frozen=True, slots=True)
class DatasetExampleSelection:
    dataset: DatasetName
    scene_id: str
    frame_ids: tuple[str, str]


def _example_frame_candidates(
    *,
    dataset: DatasetName,
    dataset_root: Path,
    scenes: list[str],
) -> list[DatasetExampleSelection]:
    from src.misc.frame_layout import FramePaths, list_frame_ids

    candidates: list[DatasetExampleSelection] = []
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
            if not (
                paths_a.image.is_file()
                and paths_a.label.is_file()
                and paths_b.image.is_file()
                and paths_b.label.is_file()
            ):
                continue
            candidates.append(
                DatasetExampleSelection(
                    dataset=dataset,
                    scene_id=scene_id,
                    frame_ids=(frame_a, frame_b),
                )
            )
    return candidates


def _pick_example_selections(
    *,
    replica_root: Path,
    scannet_root: Path,
    seed: int,
) -> dict[DatasetName, DatasetExampleSelection]:
    rng = random.Random(seed)
    selections: dict[DatasetName, DatasetExampleSelection] = {}

    for dataset, dataset_root in (
        ("scannet", scannet_root),
        ("replica", replica_root),
    ):
        candidates = _example_frame_candidates(
            dataset=dataset,  # type: ignore[arg-type]
            dataset_root=dataset_root,
            scenes=TEST_SCENES[dataset],  # type: ignore[index]
        )
        if not candidates:
            raise FileNotFoundError(
                f"No consecutive labeled RGB frames found under {dataset_root}."
            )
        selections[dataset] = rng.choice(candidates)  # type: ignore[index]

    return selections


def _collect_class_ids(
    *,
    selections: dict[DatasetName, DatasetExampleSelection],
    dataset_roots: dict[DatasetName, Path],
) -> set[int]:
    import numpy as np

    from src.misc.frame_layout import FramePaths

    class_ids: set[int] = set()
    for dataset, selection in selections.items():
        scene_dir = dataset_roots[dataset] / selection.scene_id
        for frame_id in selection.frame_ids:
            label_path = FramePaths.from_frame_id(scene_dir, frame_id).label
            gt_dense = _load_gt_dense(label_path)
            class_ids.update(int(class_id) for class_id in np.unique(gt_dense) if class_id)
    class_ids.discard(0)
    return class_ids


def _load_rgb_image(image_path: Path):
    from PIL import Image

    return Image.open(image_path).convert("RGB")


def _cell_size(reference_images, *, max_cell_width: int) -> tuple[int, int]:
    from PIL import Image

    reference = max(reference_images, key=lambda image: image.width * image.height)
    width, height = reference.size
    if width > max_cell_width:
        scale = max_cell_width / width
        width = max_cell_width
        height = max(1, round(height * scale))
    return width, height


def _prepare_cell(image, target_size: tuple[int, int]):
    return _crop_center_to_size(image, target_size[0], target_size[1])


def _compose_data_examples_figure(
    *,
    selections: dict[DatasetName, DatasetExampleSelection],
    dataset_roots: dict[DatasetName, Path],
    colormap: dict[int, tuple[int, int, int]],
    output_path: Path,
    max_cell_width: int = 320,
) -> None:
    from PIL import Image, ImageDraw

    from src.misc.frame_layout import FramePaths

    header_font = _load_table_font(TABLE_HEADER_FONT_SIZE)
    column_gap = 2
    row_gap = 2
    margin = 2
    header_height = TABLE_HEADER_FONT_SIZE + TABLE_HEADER_PADDING

    image_cells: list[Image.Image] = []
    mask_cells: list[Image.Image] = []
    for dataset, _label in DATASET_COLUMNS:
        selection = selections[dataset]
        scene_dir = dataset_roots[dataset] / selection.scene_id
        for frame_id in selection.frame_ids:
            paths = FramePaths.from_frame_id(scene_dir, frame_id)
            image_cells.append(_load_rgb_image(paths.image))
            mask_rgb = _colorize_dense_mask(_load_gt_dense(paths.label), colormap)
            mask_cells.append(Image.fromarray(mask_rgb))

    cell_width, cell_height = _cell_size(image_cells + mask_cells, max_cell_width=max_cell_width)
    image_cells = [_prepare_cell(image, (cell_width, cell_height)) for image in image_cells]
    mask_cells = [_prepare_cell(image, (cell_width, cell_height)) for image in mask_cells]

    body_width = (
        NUM_COLUMNS * cell_width + column_gap * (NUM_COLUMNS - 1)
    )
    body_height = NUM_ROWS * cell_height + row_gap * (NUM_ROWS - 1)
    canvas_width = margin * 2 + body_width
    canvas_height = margin * 2 + header_height + row_gap + body_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    body_left = margin
    header_top = margin
    group_width = EXAMPLES_PER_DATASET * cell_width + column_gap * (EXAMPLES_PER_DATASET - 1)
    for group_index, (_dataset, label) in enumerate(DATASET_COLUMNS):
        group_left = body_left + group_index * EXAMPLES_PER_DATASET * (cell_width + column_gap)
        _draw_centered_text(
            draw,
            (
                group_left,
                header_top,
                group_left + group_width,
                header_top + header_height,
            ),
            label,
            font=header_font,
        )

    y = margin + header_height + row_gap
    for column_index, image in enumerate(image_cells):
        x = body_left + column_index * (cell_width + column_gap)
        canvas.paste(image, (x, y))

    y += cell_height + row_gap
    for column_index, image in enumerate(mask_cells):
        x = body_left + column_index * (cell_width + column_gap)
        canvas.paste(image, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def generate_data_examples_figure(
    *,
    replica_root: str | Path,
    scannet_root: str | Path,
    output_path: str | Path,
    seed: int = DEFAULT_SEED,
    max_cell_width: int = 320,
) -> dict:
    replica_path = Path(replica_root)
    scannet_path = Path(scannet_root)
    out_path = Path(output_path)

    selections = _pick_example_selections(
        replica_root=replica_path,
        scannet_root=scannet_path,
        seed=seed,
    )
    dataset_roots = {"replica": replica_path, "scannet": scannet_path}
    colormap = _build_class_colormap(
        _collect_class_ids(selections=selections, dataset_roots=dataset_roots)
    )
    _compose_data_examples_figure(
        selections=selections,
        dataset_roots=dataset_roots,
        colormap=colormap,
        output_path=out_path,
        max_cell_width=max_cell_width,
    )

    payload = {
        "output_path": str(out_path),
        "seed": seed,
        "selections": {
            dataset: {
                "scene_id": selection.scene_id,
                "frame_a": selection.frame_ids[0],
                "frame_b": selection.frame_ids[1],
            }
            for dataset, selection in selections.items()
        },
    }
    print(json.dumps(payload, indent=2))
    return payload


def _attach_figure_bytes(report: dict, output_path: Path) -> dict:
    report["figure_bytes"] = {output_path.name: output_path.read_bytes()}
    return report


@app.function(
    image=viz_image,
    cpu=2,
    memory=4096,
    timeout=60 * 15,
    volumes=DATA_EXAMPLES_VOLUMES,
    nonpreemptible=True,
)
def render_data_examples(
    replica_root: str | None = None,
    scannet_root: str | None = None,
    seed: int = DEFAULT_SEED,
    max_cell_width: int = 320,
) -> dict:
    replica_path = Path(resolve_dataset_root("replica", replica_root))
    scannet_path = Path(resolve_dataset_root("scannet", scannet_root))
    for dataset_name, root in (("replica", replica_path), ("scannet", scannet_path)):
        if not root.is_dir():
            raise FileNotFoundError(
                f"{dataset_name} dataset not found at {root}. "
                f"Populate the `{dataset_name}` volume."
            )

    out_path = REMOTE_SCRATCH_DIR / "data_examples.png"
    report = generate_data_examples_figure(
        replica_root=replica_path,
        scannet_root=scannet_path,
        output_path=out_path,
        seed=seed,
        max_cell_width=max_cell_width,
    )
    return _attach_figure_bytes(report, out_path)


def save_figure_locally(report: dict, output_path: Path | None = None) -> Path:
    out_path = (output_path or default_output_path()).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for _name, data in report.get("figure_bytes", {}).items():
        out_path.write_bytes(data)
    print(f"Wrote {out_path}")
    return out_path


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
    output_path: str | None = None,
    seed: int = DEFAULT_SEED,
    max_cell_width: int = 320,
    detach: bool | None = None,
    wait: bool = False,
) -> None:
    """Render dataset examples from Modal volumes."""
    detached = resolve_detach(detach=detach, remote_job=not wait)
    report = _dispatch(
        render_data_examples,
        job_name="data examples",
        detach=detached,
        replica_root=replica_root,
        scannet_root=scannet_root,
        seed=seed,
        max_cell_width=max_cell_width,
    )
    if detached:
        print(
            "Detached run: figure is not saved locally. "
            "Re-run with --wait to write into c3gsam_results/data_examples.png."
        )
        return

    local_out = Path(output_path) if output_path else default_output_path()
    save_figure_locally(report, local_out)


def _parse_local_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render RGB + GT mask dataset examples (2×4 figure)."
    )
    parser.add_argument("--replica-root", type=Path, default=None)
    parser.add_argument("--scannet-root", type=Path, default=None)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=default_output_path(),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-cell-width",
        type=int,
        default=320,
        help="Maximum width of each panel before layout (default: 320).",
    )
    return parser.parse_args(argv)


def main_local(argv: list[str] | None = None) -> None:
    args = _parse_local_args(argv)
    generate_data_examples_figure(
        replica_root=args.replica_root or default_local_dataset_root("replica"),
        scannet_root=args.scannet_root or default_local_dataset_root("scannet"),
        output_path=args.output_path,
        seed=args.seed,
        max_cell_width=args.max_cell_width,
    )


if __name__ == "__main__":
    main_local(sys.argv[1:])
