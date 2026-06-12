# Distillation Training with Pre-computed SAM Features

**Form 1** — train C3G-SAM to match offline SAM ViT-H encoder features. Canonical entry-point reference: **[12-c3g-sam.md](12-c3g-sam.md)**.

Unlike prompted training (Form 2), distillation pre-computes SAM features offline and trains without loading SAM into GPU memory.

## Entry points

| Environment | Command |
|-------------|---------|
| **Local precompute** | `uv run python scripts/precompute_sam_features.py --dataset-root … --dataset replica\|scannet --sam-checkpoint …` |
| **Modal precompute** | `modal run src/modal/precompute.py::main --dataset scannet --wait` |
| **Local train** | `uv run python -m src.main +training=feature_head_sam_precomputed …` |
| **Modal train** | `modal run src/modal/train.py --wait` |
| **Modal smoke** | `modal run src/modal/train.py::smoke --wait` |

Hydra preset **`feature_head_sam_precomputed`** targets ScanNet on Modal (`scannet_distill`, mounts `/scannet` and `/precompute_sam_features/scannet`). Override `dataset.scannet_distill.roots` for local paths.

## Prerequisites

### 1. Environment Setup

```bash
uv sync --frozen
source .venv/bin/activate
```

Set `NUM_SEMANTIC_CHANNELS=256` in `submodules/diff_gaussian_rasterization_w_feature_detach/cuda_rasterizer/config.h` and reinstall the submodule.

### 2. SAM Checkpoint (for pre-computation only)

```bash
mkdir -p pretrained_weights/
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O ./pretrained_weights/sam_vit_h.pth
```

Not loaded during distillation training — only for precompute.

### 3. Encoder Pretrained Weights

```bash
wget https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt?download=true -O ./pretrained_weights/model.pt
# Or use gaussian_decoder.ckpt as model.encoder.pretrained_weights (recommended for SAM)
```

### 4. Dataset

Flat per-scene layout under `datasets/replica` or `datasets/scannet`:

```
datasets/scannet/
├── scene0000_00/
│   ├── 00000_x.jpg
│   ├── 00000_y.png
│   ├── 00000_cam.npz
│   └── ...
```

Replica scenes: `office0`–`office4`, `room0`–`room2`. Prepare with `python -m src.dataset.download_replica` or Modal `download_replica.py`.

## Step 1: Pre-compute SAM Features

Produces `{frame_id}_sam.pt` (256×64×64 float32) beside each frame.

```bash
uv run python scripts/precompute_sam_features.py \
    --dataset-root datasets/scannet \
    --dataset scannet \
    --sam-checkpoint pretrained_weights/sam_vit_h.pth \
    --batch-size 8
```

On Modal, features land on volume **`precompute_sam_features`** at `scannet/<scene>/` or `replica/<scene>/`.

### Pre-computation Arguments

| Argument             | Default      | Description                                      |
| -------------------- | ------------ | ------------------------------------------------ |
| `--dataset-root`     | (required)   | Root directory of the dataset                    |
| `--dataset`          | (required)   | Dataset type: `replica` or `scannet`             |
| `--scenes`           | all scenes   | Specific scenes to process                       |
| `--sam-checkpoint`   | (required)   | Path to SAM checkpoint file                      |
| `--sam-model-variant`| `sam_vit_h`  | SAM model variant                                |
| `--batch-size`       | `8`          | Frames per encoding batch                        |
| `--overwrite`        | `false`      | Overwrite existing `.pt` files                   |

## Step 2: Run Distillation Training

**Local (ScanNet example):**

```bash
uv run python -m src.main +training=feature_head_sam_precomputed \
    dataset.scannet_distill.roots=[./datasets/scannet] \
    dataset.scannet_distill.sam_features_root=./datasets/scannet \
    model.encoder.pretrained_weights=./pretrained_weights/gaussian_decoder.ckpt \
    wandb.mode=disabled
```

**Modal** (uses paths from YAML — no CLI overrides):

```bash
modal run src/modal/train.py --wait
```

Checkpoints: Modal volume **`c3g-train-outputs`** → `/outputs/runs/sam_distill_scannet/checkpoints/`.

### Key Training Parameters

| Parameter                          | Default  | Description                                         |
| ---------------------------------- | -------- | --------------------------------------------------- |
| `train.pipeline`                   | `distillation` | Selects the distillation training loop        |
| `train.feature_cosine_loss_weight` | `1.0`    | Weight for cosine feature loss                      |
| `train.feature_mag_loss_weight`    | `0.5`    | Weight for feature magnitude MSE                    |
| `train.context_view_loss`          | `true`   | Include context views in the feature loss           |
| `optimizer.lr`                     | `1.5e-5` | Learning rate                                       |
| `trainer.max_steps`                | `5001`   | Total training steps                                |
| `data_loader.train.batch_size`     | `6`      | Batch size (ScanNet Modal default)                  |

### Frozen Components

- `freeze_backbone: true` — VGGT backbone frozen
- `freeze_instill_qk: true` — Cross-attention Q/K frozen
- `freeze_geometry_head: true` — Gaussian geometry head frozen
- `feature_detach: true` — Geometry detached from feature loss gradient

## Evaluation

Copy a trained checkpoint to **`c3g-weights`** as `distillation-base.ckpt`, then:

```bash
modal run src/modal/eval_masks.py::c3g --wait
modal run src/modal/get_scores.py --experiment c3gsam --wait
```

Local export:

```bash
uv run python -m src.evaluation.mask_export \
    +evaluation=c3g_sam_distill checkpointing.load=path/to/checkpoint.ckpt
```

## Expected Outputs

### Checkpoints

Local: `outputs/<date>/<time>/checkpoints/`. Modal: `/outputs/runs/<wandb.name>/checkpoints/`.

Config saves top 20 checkpoints every 50 steps.

### Wandb Monitoring

- `loss/feature_cosine`, `loss/feature_mag`, `loss/total`
- `val/feature_cosine`, `val/feature_mag`

## How It Works

1. `DatasetScannetDistill` / `DatasetReplicaDistill` load `{frame_id}_sam.pt` alongside RGB and cameras.
2. Encoder produces 2048 Gaussians with 256-dim features from context views.
3. Decoder renders feature maps; bilinear resize to 64×64 SAM resolution.
4. Cosine + magnitude losses vs precomputed SAM features.

## Comparison with Prompted Training (Form 2)

| Aspect                  | Form 2 — Prompted                    | Form 1 — Distillation                |
| ----------------------- | ------------------------------------ | ------------------------------------ |
| SAM at train time       | Yes (full model in GPU memory)       | No (pre-computed offline)            |
| Loss function           | BCE + Dice on segmentation masks     | Cosine + magnitude on encoder features |
| Config                  | `feature_head_sam_prompted_scannet`  | `feature_head_sam_precomputed`       |
| Modal                   | `--experiment prompted`              | default `distillation`               |

See [prompted_training.md](prompted_training.md) for Form 2.
