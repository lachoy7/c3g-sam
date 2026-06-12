"""Train / val / test scene splits for the Modal ``scannet`` volume (807 scenes).

Scenes are ordered by scan number (``scene0000_00`` … ``scene0806_00``). The
highest-numbered scenes are held out:

- **test**: last 80 scenes (``scene0727_00`` … ``scene0806_00``)
- **val**: 80 scenes immediately before test (``scene0647_00`` … ``scene0726_00``)
- **train**: all earlier scenes (647 scenes)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

Stage = Literal["train", "val", "test"]

NUM_VAL_SCENES = 80
NUM_TEST_SCENES = 80
SCENE_ID_TEMPLATE = "scene{index:04d}_00"
ALL_SCENE_IDS = tuple(SCENE_ID_TEMPLATE.format(index=i) for i in range(807))


def scene_sort_key(scene_id: str) -> int:
    """Numeric index from a ScanNet scene id (e.g. ``scene0697_00`` -> 697)."""
    return int(scene_id[5:9])


def sort_scene_ids(scene_ids: list[str]) -> list[str]:
    return sorted(scene_ids, key=scene_sort_key)


def discover_scene_ids(root: Path) -> list[str]:
    """List prepared scene directories under ``root``, sorted by scan number."""
    if not root.is_dir():
        return []
    scene_ids = [
        path.name
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("scene") and "_" in path.name
    ]
    return sort_scene_ids(scene_ids)


def split_scene_ids(
    scene_ids: list[str],
    *,
    num_val: int = NUM_VAL_SCENES,
    num_test: int = NUM_TEST_SCENES,
) -> tuple[list[str], list[str], list[str]]:
    """Partition sorted ``scene_ids`` into train, val, and test lists."""
    ordered = sort_scene_ids(scene_ids)
    n = len(ordered)
    if n < num_val + num_test:
        raise ValueError(
            f"Need at least {num_val + num_test} scenes for val+test splits, got {n}"
        )
    train = ordered[: n - num_test - num_val]
    val = ordered[n - num_test - num_val : n - num_test]
    test = ordered[n - num_test :]
    return train, val, test


def scenes_for_stage(
    stage: Stage,
    *,
    scene_ids: list[str] | None = None,
    root: Path | None = None,
    num_val: int = NUM_VAL_SCENES,
    num_test: int = NUM_TEST_SCENES,
) -> list[str]:
    """Return scene ids for a Lightning stage."""
    if scene_ids is None:
        if root is None:
            scene_ids = list(ALL_SCENE_IDS)
        else:
            scene_ids = discover_scene_ids(root)
    train, val, test = split_scene_ids(
        scene_ids, num_val=num_val, num_test=num_test
    )
    if stage == "train":
        return train
    if stage == "val":
        return val
    return test


# Canonical splits for the full 807-scene volume (used in docs and smoke tests).
SCENES_TRAIN, SCENES_VAL, SCENES_TEST = split_scene_ids(list(ALL_SCENE_IDS))
