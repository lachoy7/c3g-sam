import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers.wandb import WandbLogger
from matplotlib import cm

from src.misc.utils import inverse_normalize
from src.model.utils import run_pca


def log_debug_visualizations(
    logger,
    global_step,
    checkpoint_interval,
    target_rgb,
    target_sam_feature,
    rendered_feature,
    img_size,
    prefix="val",
    valid_region=None,
    rendered_rgb=None,
):
    """Log debug visualization table and feature norm stats to wandb.

    SAM features live in a padded-square embedding grid where only the
    top-left ``valid_region`` holds real content; the rest is zero padding.
    Crop to that region before PCA / heatmaps so the overlay aligns with the
    (full-frame) RGB image instead of being squashed against the padding seam.
    """
    if global_step % checkpoint_interval != 0:
        return
    if not isinstance(logger, WandbLogger):
        return
    if rendered_feature is None:
        return

    h, w = img_size
    target_rgb_norm = inverse_normalize(target_rgb)

    if valid_region is not None:
        valid_h, valid_w = valid_region
        target_sam_feature = target_sam_feature[:, :valid_h, :valid_w]
        rendered_feature = rendered_feature[:, :valid_h, :valid_w]

    target_pca = run_pca(target_sam_feature.unsqueeze(0), (h, w))
    rendered_pca = run_pca(rendered_feature.unsqueeze(0), (h, w))

    mse_map = compute_mse_heatmap(rendered_feature, target_sam_feature)
    cosine_map = compute_cosine_error_map(rendered_feature, target_sam_feature)

    mse_resized = (
        F.interpolate(
            mse_map.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
    )
    cosine_resized = (
        F.interpolate(
            cosine_map.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
    )

    mse_colored = colorize_heatmap(mse_resized)
    cosine_colored = colorize_heatmap(cosine_resized)

    mse_overlay = alpha_blend(mse_colored, target_rgb_norm)
    cosine_overlay = alpha_blend(cosine_colored, target_rgb_norm)

    columns = [
        "target_rgb",
        "target_sam_pca",
        "rendered_feature_pca",
        "mse_heatmap_overlay",
        "cosine_error_overlay",
    ]
    if rendered_rgb is not None:
        columns.append("rendered_rgb")

    table = wandb.Table(columns=columns)

    row = [
        wandb.Image(to_hwc_uint8(target_rgb_norm)),
        wandb.Image(to_hwc_uint8(target_pca.squeeze(0))),
        wandb.Image(to_hwc_uint8(rendered_pca.squeeze(0))),
        wandb.Image(to_hwc_uint8(mse_overlay)),
        wandb.Image(to_hwc_uint8(cosine_overlay)),
    ]
    if rendered_rgb is not None:
        row.append(wandb.Image(to_hwc_uint8(rendered_rgb)))

    table.add_data(*row)
    logger.experiment.log({f"{prefix}/debug_visualizations": table})

    target_norms = compute_feature_norms(target_sam_feature)
    rendered_norms = compute_feature_norms(rendered_feature)

    target_norms_np = target_norms.detach().cpu().numpy()
    rendered_norms_np = rendered_norms.detach().cpu().numpy()

    logger.experiment.log(
        {
            f"{prefix}/target_feature_norms": wandb.Histogram(target_norms_np),
            f"{prefix}/rendered_feature_norms": wandb.Histogram(rendered_norms_np),
        },
    )

    stats_table = wandb.Table(
        columns=["source", "min", "max", "mean", "std"],
        data=[
            [
                "target",
                target_norms.min().item(),
                target_norms.max().item(),
                target_norms.mean().item(),
                target_norms.std().item(),
            ],
            [
                "rendered",
                rendered_norms.min().item(),
                rendered_norms.max().item(),
                rendered_norms.mean().item(),
                rendered_norms.std().item(),
            ],
        ],
    )
    logger.experiment.log({f"{prefix}/feature_norm_stats": stats_table})


def to_hwc_uint8(tensor):
    """Convert (3, H, W) float tensor in [0, 1] to HWC uint8 numpy array."""
    return (
        (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0) * 255)
        .to(torch.uint8)
        .numpy()
    )


def compute_mse_heatmap(rendered_feature, target_feature):
    """Compute per-pixel MSE across channels, returns (H, W) tensor."""
    return ((rendered_feature - target_feature) ** 2).mean(dim=0)


def compute_cosine_error_map(rendered_feature, target_feature):
    """Compute 1 - cosine_similarity per pixel, returns (H, W) tensor."""
    eps = 1e-8
    rendered_norm = rendered_feature.norm(dim=0, keepdim=True).clamp(min=eps)
    target_norm = target_feature.norm(dim=0, keepdim=True).clamp(min=eps)
    rendered_normalized = rendered_feature / rendered_norm
    target_normalized = target_feature / target_norm
    cosine_similarity = (rendered_normalized * target_normalized).sum(dim=0)
    return 1 - cosine_similarity


def colorize_heatmap(heatmap, vmin=0.0, vmax=None):
    """Apply viridis colormap to a heatmap, returns (3, H, W) in [0, 1]."""
    if vmax is None:
        vmax = heatmap.max()
    normalized = ((heatmap - vmin) / (vmax - vmin + 1e-8)).clamp(0, 1)
    cmap = cm.get_cmap("viridis")
    rgb_np = cmap(normalized.detach().cpu().numpy())[..., :3]
    rgb = torch.tensor(rgb_np, device=heatmap.device, dtype=torch.float32)
    return rgb.permute(2, 0, 1)


def alpha_blend(overlay, background, alpha=0.5):
    """Blend overlay onto background, both (3, H, W) in [0, 1], returns (3, H, W)."""
    return alpha * overlay + (1 - alpha) * background


def compute_feature_norms(feature):
    """Compute per-spatial-position L2 norms, returns flat vector of length H*W."""
    return feature.norm(dim=0).flatten()


def crop_masks_to_valid(masks, embed_size, valid_region):
    """Crop SAM decoder masks to the valid (non-padded) region.

    SAM features are computed from images padded to 1024x1024. The mask decoder
    outputs (B, N, 256, 256) masks in that padded space. ``valid_region`` is the
    valid extent in the ``embed_size`` (e.g. 64x64) embedding grid, which we
    scale into the 256x256 mask space. This is applied to both the SAM and the
    rendered masks so they stay consistent (the rendered embeddings have no zero
    padding to auto-detect, so an explicit region is required).
    """
    _, _, mask_h, mask_w = masks.shape
    valid_h, valid_w = valid_region

    crop_h = int(round(valid_h / embed_size * mask_h))
    crop_w = int(round(valid_w / embed_size * mask_w))
    crop_h = min(crop_h, mask_h)
    crop_w = min(crop_w, mask_w)

    if crop_h == mask_h and crop_w == mask_w:
        return masks
    return masks[:, :, :crop_h, :crop_w]


def log_decoder_debug(
    logger,
    global_step,
    checkpoint_interval,
    sam_decoder,
    target_sam_feature,
    rendered_feature,
    target_rgb,
    img_size,
    prefix="train",
    valid_region=None,
):
    """Run SAM decoder on both SAM and rendered embeddings, log masks to wandb table.

    The decoder operates on the full padded-square (64x64) embeddings, as SAM
    expects, but its output masks are cropped to ``valid_region`` so they align
    with the full-frame RGB image.
    """
    if global_step % checkpoint_interval != 0:
        return
    if not isinstance(logger, WandbLogger):
        return
    if rendered_feature is None or sam_decoder is None:
        return

    h, w = img_size
    target_rgb_norm = inverse_normalize(target_rgb)

    embed_size = 64
    with torch.no_grad():
        sam_input = target_sam_feature.unsqueeze(0)
        rendered_input = rendered_feature.unsqueeze(0)

        if sam_input.shape[-2:] != (embed_size, embed_size):
            sam_input = F.interpolate(
                sam_input,
                size=(embed_size, embed_size),
                mode="bilinear",
                align_corners=False,
            )
        if rendered_input.shape[-2:] != (embed_size, embed_size):
            rendered_input = F.interpolate(
                rendered_input,
                size=(embed_size, embed_size),
                mode="bilinear",
                align_corners=False,
            )

        sam_masks = sam_decoder(sam_input)
        rendered_masks = sam_decoder(rendered_input)

        if valid_region is not None:
            sam_masks = crop_masks_to_valid(sam_masks, embed_size, valid_region)
            rendered_masks = crop_masks_to_valid(
                rendered_masks, embed_size, valid_region
            )

    sam_overlay = colorize_masks_overlay(sam_masks[0], target_rgb_norm, (h, w))
    rendered_overlay = colorize_masks_overlay(
        rendered_masks[0], target_rgb_norm, (h, w)
    )

    num_masks = sam_masks.shape[1]
    columns = ["target_rgb", "sam_masks_overlay", "rendered_masks_overlay"]
    for i in range(num_masks):
        columns.append(f"sam_mask_{i}")
        columns.append(f"rendered_mask_{i}")

    table = wandb.Table(columns=columns)

    row = [
        wandb.Image(to_hwc_uint8(target_rgb_norm)),
        wandb.Image(to_hwc_uint8(sam_overlay)),
        wandb.Image(to_hwc_uint8(rendered_overlay)),
    ]

    for i in range(num_masks):
        sam_mask_i = sam_masks[0, i]
        rendered_mask_i = rendered_masks[0, i]

        sam_mask_resized = (
            F.interpolate(
                sam_mask_i.unsqueeze(0).unsqueeze(0).float(),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

        rendered_mask_resized = (
            F.interpolate(
                rendered_mask_i.unsqueeze(0).unsqueeze(0).float(),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

        sam_mask_rgb = mask_to_colored_rgb(sam_mask_resized, i)
        rendered_mask_rgb = mask_to_colored_rgb(rendered_mask_resized, i)

        row.append(wandb.Image(to_hwc_uint8(sam_mask_rgb)))
        row.append(wandb.Image(to_hwc_uint8(rendered_mask_rgb)))

    table.add_data(*row)
    logger.experiment.log({f"{prefix}/decoder_debug": table})


MASK_COLORS = [
    torch.tensor([1.0, 0.2, 0.2]),  # red
    torch.tensor([0.2, 0.8, 0.2]),  # green
    torch.tensor([0.2, 0.4, 1.0]),  # blue
    torch.tensor([1.0, 0.8, 0.0]),  # yellow
]


def mask_to_colored_rgb(mask_logits, mask_idx):
    """Convert mask logits to a colored RGB visualization (3, H, W) in [0, 1]."""
    prob = torch.sigmoid(mask_logits)
    color = MASK_COLORS[mask_idx % len(MASK_COLORS)].to(mask_logits.device)
    return prob.unsqueeze(0) * color.view(3, 1, 1)


def colorize_masks_overlay(masks, background, img_size):
    """Blend all masks as colored overlays onto background image, returns (3, H, W)."""
    h, w = img_size
    num_masks = masks.shape[0]
    overlay = torch.zeros(3, h, w, device=masks.device)

    for i in range(num_masks):
        mask_resized = (
            F.interpolate(
                masks[i].unsqueeze(0).unsqueeze(0).float(),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )
        prob = torch.sigmoid(mask_resized)
        color = MASK_COLORS[i % len(MASK_COLORS)].to(masks.device)
        overlay += prob.unsqueeze(0) * color.view(3, 1, 1)

    overlay = overlay.clamp(0, 1)
    return alpha_blend(overlay, background, alpha=0.5)
