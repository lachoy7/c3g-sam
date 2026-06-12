"""Generates point prompts from GT binary masks for prompted SAM training."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from src.model.sam.constants import SAM_IMAGE_SIZE

CoordFrame = Literal["sam", "original"]


def decompose_label_map(label_map: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Label map (H, W) -> (K, H, W) binary masks for non-background class ids."""
    if isinstance(label_map, torch.Tensor):
        label_np = label_map.detach().cpu().numpy()
    else:
        label_np = np.asarray(label_map)

    unique_ids = np.unique(label_np)
    non_bg_ids = unique_ids[unique_ids != 0]
    if len(non_bg_ids) == 0:
        h, w = label_np.shape
        return torch.zeros((0, h, w), dtype=torch.float32)

    masks = [(label_np == obj_id).astype(np.float32) for obj_id in non_bg_ids]
    return torch.from_numpy(np.stack(masks, axis=0))


class PromptSampler:
    """Generates point prompts from GT binary masks for prompted SAM training."""

    def __init__(
        self, strategy="centroid", min_object_pixels=16, image_size=SAM_IMAGE_SIZE
    ):
        self.strategy = strategy
        self.min_object_pixels = min_object_pixels
        self.image_size = image_size

    def sample(
        self,
        binary_masks: torch.Tensor,
        *,
        coord_frame: CoordFrame = "sam",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pick a random valid mask and return one foreground point prompt.

        ``coord_frame="sam"`` maps (row, col) to the SAM longest-side canvas (for the
        C3G feature decoder). ``coord_frame="original"`` keeps pixel coords in the
        label/image frame (for vanilla SAM ``forward``).
        """
        K, H, W = binary_masks.shape
        valid_indices = []
        for i in range(K):
            if binary_masks[i].sum().item() >= self.min_object_pixels:
                valid_indices.append(i)

        if len(valid_indices) == 0:
            raise ValueError("No mask has enough foreground pixels for prompt sampling")

        idx = valid_indices[torch.randint(len(valid_indices), size=()).item()]
        selected_mask = binary_masks[idx]

        if self.strategy == "centroid":
            row, col = self.compute_centroid(selected_mask)
        else:
            row, col = self.sample_random_point(selected_mask)

        if coord_frame == "original":
            x, y = float(col), float(row)
        else:
            x = col * (self.image_size / W)
            y = row * (self.image_size / H)

        point_coords = torch.tensor([[x, y]], dtype=torch.float32)
        point_labels = torch.tensor([1], dtype=torch.int64)

        return point_coords, point_labels, selected_mask

    def compute_centroid(self, mask):
        """Mean row/col of foreground pixels, rounded to nearest int."""
        fg = torch.nonzero(mask, as_tuple=False)
        mean_row = fg[:, 0].float().mean().round().long().item()
        mean_col = fg[:, 1].float().mean().round().long().item()
        return mean_row, mean_col

    def sample_random_point(self, mask):
        """Uniform random foreground pixel."""
        fg = torch.nonzero(mask, as_tuple=False)
        idx = torch.randint(fg.shape[0], size=()).item()
        row = fg[idx, 0].item()
        col = fg[idx, 1].item()
        return row, col
