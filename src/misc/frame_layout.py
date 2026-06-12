"""On-disk layout for C3G 2D segmentation scenes.

Each scene directory holds one triplet per frame::

    {frame_id:05d}_x.jpg   — RGB image
    {frame_id:05d}_cam.npz — ``camera_pose`` (4×4), ``camera_intrinsics`` (3×3)
    {frame_id:05d}_y.png   — semantic label map (uint16; PNG preserves class ids)

``frame_id`` is zero-padded to five digits (e.g. ``00042``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

FRAME_ID_WIDTH = 5


def format_frame_id(frame_index: int) -> str:
    """Return the canonical zero-padded frame id string."""
    return f"{frame_index:0{FRAME_ID_WIDTH}d}"


@dataclass(frozen=True, slots=True)
class FramePaths:
    """Paths for a single frame's image, camera, and label files."""

    frame_id: str
    image: Path
    camera: Path
    label: Path

    @classmethod
    def from_index(cls, scene_dir: Path, frame_index: int) -> FramePaths:
        return cls.from_frame_id(scene_dir, format_frame_id(frame_index))

    @classmethod
    def from_frame_id(cls, scene_dir: Path, frame_id: str) -> FramePaths:
        return cls(
            frame_id=frame_id,
            image=scene_dir / f"{frame_id}_x.jpg",
            camera=scene_dir / f"{frame_id}_cam.npz",
            label=scene_dir / f"{frame_id}_y.png",
        )


def list_frame_ids(scene_dir: Path) -> list[str]:
    """List frame ids present in ``scene_dir``, sorted lexicographically."""
    suffix = "_x.jpg"
    return sorted(
        path.name[: -len(suffix)]
        for path in scene_dir.glob(f"*{suffix}")
        if path.is_file()
    )
