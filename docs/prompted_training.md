# Prompted Training with SAM on Replica SemSeg

**Form 2** — point-prompted SAM segmentation. Entry-point hub: **[12-c3g-sam.md](12-c3g-sam.md)**. Modal / ScanNet: [prompted_training_modal.md](prompted_training_modal.md).

This document describes local prompted training on the flat **Replica 2D-seg** layout (`replica_2dseg` dataset).

## Prerequisites

### 1. Environment Setup

```bash
uv sync --frozen
source .venv/bin/activate
```

### 2. SAM Checkpoint

Download the SAM ViT-H checkpoint into `pretrained_weights/`:

```bash
mkdir pretrained_weights/
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O ./pretrained_weights/sam_vit_h.pth
```

### 3. Encoder Pretrained Weights

Download the VGGT pretrained weights (if not already present):

```bash
wget https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt?download=true -O ./pretrained_weights/model.pt
```

### 4. Dataset

Use the flat **2D-seg** layout (`{frame_id}_x.jpg`, `{frame_id}_y.png`, `{frame_id}_cam.npz`). Default local root: `datasets/replica`. Prepare with:

```bash
python -m src.dataset.download_replica --source /path/to/raw/replica --out-dir ./datasets/replica
```

Legacy Replica SemSeg tree (`replica_semseg`) is still supported via `dataset.replica_semseg` but **`replica_2dseg` is recommended**.

## Training

Run prompted-mode training:

```bash
uv run python -m src.main +training=feature_head_sam_prompted \
    +dataset@_group_.replica_2dseg=replica_2dseg \
    wandb.mode=online wandb.name="sam_prompted"
```

To override the dataset root:

```bash
uv run python -m src.main +training=feature_head_sam_prompted \
    +dataset@_group_.replica_2dseg=replica_2dseg \
    wandb.mode=online \
    wandb.name="sam_prompted" \
    dataset.replica_2dseg.roots="[/path/to/replica]"
```

To use random-point prompts instead of centroid:

```bash
uv run python -m src.main +training=feature_head_sam_prompted \
    wandb.mode=online \
    wandb.name="sam_prompted_random" \
    train.prompt_strategy=random_point \
    dataset.replica_semseg.prompt_strategy=random_point
```

To disable wandb logging:

```bash
uv run python -m src.main +training=feature_head_sam_prompted wandb.mode=disabled
```

### Key Training Parameters


| Parameter                        | Default     | Description                                |
| -------------------------------- | ----------- | ------------------------------------------ |
| `train.prompt_mode`              | `prompted`  | Use point prompts from GT labels           |
| `train.prompted_seg_loss_weight` | `1.0`       | Weight for the prompted segmentation loss  |
| `train.prompt_strategy`          | `centroid`  | `centroid` or `random_point`               |
| `train.min_object_pixels`        | `16`        | Minimum foreground pixels for valid prompt |
| `train.sam_model_variant`        | `sam_vit_h` | SAM model variant                          |
| `train.use_lora`                 | `false`     | Enable LoRA adaptation on SAM decoder      |
| `train.lora_rank`                | `4`         | LoRA rank (if enabled)                     |


## Evaluation (Segment-Everything Mode)

Evaluation always uses grid prompts (segment-everything mode) regardless of the training `prompt_mode`. This produces multi-view consistent masks for qualitative and quantitative assessment.

```bash
uv run python -m src.main +training=feature_head_sam_prompted \
    mode=test \
    wandb.mode=online \
    wandb.name="sam_prompted_eval" \
    checkpointing.load="path/to/checkpoint.ckpt"
```

## Expected Outputs

### Checkpoints

Checkpoints are saved to the Hydra output directory under `checkpoints/`:

```
outputs/<date>/<time>/checkpoints/
```

The config saves the top 5 checkpoints every 10,000 steps (configurable via `checkpointing.every_n_train_steps` and `checkpointing.save_top_k`).

### Evaluation Visualizations

When running in test mode with `test.save_compare=true`, predicted masks overlaid on RGB images are saved under the output directory per scene:

```
outputs/<date>/<time>/<scene>/seg/
```

### Wandb Monitoring

When `wandb.mode=online`, the following metrics are logged:

- `loss/prompted_segmentation` — prompted BCE + Dice loss
- `loss/total` — combined training loss
- Standard reconstruction losses (MSE, LPIPS) if enabled
- Multi-view IoU metrics during validation

