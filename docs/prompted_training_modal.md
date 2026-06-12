# Prompted Training with SAM on Replica and ScanNet (Modal)

**Form 2** — point-prompted SAM segmentation on Modal. For the full volume/checkpoint/ablation reference see **[12-c3g-sam.md](12-c3g-sam.md)**. Local Replica training: [prompted_training.md](prompted_training.md).

## Entry points

| Task | Command |
|------|---------|
| Download Replica | `modal run src/dataset/download_replica.py` |
| Download ScanNet | `modal run src/dataset/download_scannet.py --accept-tos` |
| Precompute SAM features | `modal run src/modal/precompute.py::main --dataset scannet --wait` |
| **Train Form 2 (prompted)** | `modal run src/modal/train.py --experiment prompted --wait` |
| **Train Form 1 (distillation)** | `modal run src/modal/train.py --wait` |
| Smoke (1 step) | `modal run src/modal/train.py::smoke --experiment prompted --wait` |

Hydra preset for Modal prompted training: **`feature_head_sam_prompted_scannet`** (selected automatically by `--experiment prompted`). Checkpoints saved on **`c3g-train-outputs`** at `/outputs/runs/sam_prompted_scannet/checkpoints/`, ranked by **`val/loss`**.

Train on [Modal](https://modal.com/) using the prepared **`replica`** and **`scannet`** volumes. Data layout matches `src/misc/frame_layout.py` and the download scripts `src/dataset/download_replica.py` and `src/dataset/download_scannet.py`.

## Prerequisites

### 1. Modal CLI

```bash
pip install modal
modal setup
```

### 2. Pretrained weights (`c3g-weights` volume)

Upload checkpoints to the `c3g-weights` volume (mounted at `/weights` in training jobs). Same layout as `src/modal/train.py`.

```bash
modal volume put c3g-weights /path/to/sam_vit_h.pth sam_vit_h.pth
modal volume put c3g-weights /path/to/gaussian_decoder.ckpt gaussian_decoder.ckpt
```

| File | Purpose |
|------|---------|
| `sam_vit_h.pth` | SAM ViT-H mask decoder (`train.sam_checkpoint`) |
| `gaussian_decoder.ckpt` | Encoder init (`model.encoder.pretrained_weights`) |

## Populate dataset volumes

Both datasets share the same **flat** on-disk layout: one directory per scene, frames named `{frame_id}_x.jpg`, `{frame_id}_y.png`, `{frame_id}_cam.npz`. Download scripts may also write `selected_seqs_test.json` at the volume root (optional; loaders discover scenes via `scenes` in config). ScanNet also includes `scannetv2-labels.combined.tsv`.

### Replica (`replica` volume → `/replica`)

Eight scenes: `office0`–`office4`, `room0`–`room2`. Frames are strided by 20 from the source trajectories.

```bash
modal run src/dataset/download_replica.py
modal run --detach src/dataset/download_replica.py
```

Local preparation (writes `./datasets/replica` by default):

```bash
python -m src.dataset.download_replica --source /path/to/raw/replica --out-dir ./datasets/replica
```

Point training configs at `dataset.replica_2dseg.roots=[/replica]` (or the mount path you use).

### ScanNet (`scannet` volume → `/scannet`)

The Modal ``scannet`` volume holds 807 prepared scenes (`scene0000_00` … `scene0806_00`). Training uses a fixed split by scan number (see `src/dataset/scannet_2dseg_splits.py` and `assets/scannet_2dseg_scene_splits.json`): **775 train** (`scene0000_00`–`scene0774_00`), **8 val** (`scene0775_00`–`scene0782_00`), **24 test** (`scene0783_00`–`scene0806_00`). W&B `val/*` metrics run on the val scenes only; test is held out for a separate eval script. Requires accepting the ScanNet terms of use.

```bash
modal run src/dataset/download_scannet.py --accept-tos
modal run --detach src/dataset/download_scannet.py --accept-tos
```

Local preparation:

```bash
python -m src.dataset.download_scannet --out-dir ./datasets/scannet --accept-tos
```

Point training configs at `dataset.scannet_2dseg.roots=[/scannet]`.

Use `src/misc/modal_run.py` (`--detach`) for long download jobs; raw `.sens` archives are downloaded to ephemeral scratch on Modal, not stored on the volume.

## Volume layout

```
/<volume_root>/
├── selected_seqs_test.json
├── scannetv2-labels.combined.tsv    # ScanNet only
├── <scene_id>/
│   ├── 00000_x.jpg                  # RGB
│   ├── 00000_y.png                  # semantic labels (uint16)
│   ├── 00000_cam.npz                # camera_pose (4×4), camera_intrinsics (3×3)
│   ├── 00020_x.jpg
│   └── ...
└── <scene_id>/
    └── ...
```

Loaders: `replica_2dseg` (`src/dataset/dataset_replica_2dseg.py`), `scannet_2dseg` (`src/dataset/dataset_scannet_2dseg.py`). They use the same sampling and batch layout as `dataset_replica_semseg` (random context/target pairs for train; full sweep for test).

## Training

Training runs in a CUDA image built from this repo (see `src/modal/train.py` for volume mounts and image build). Mount volumes:

| Volume | Mount | Dataset config |
|--------|-------|----------------|
| `c3g-weights` | `/weights` | SAM + encoder checkpoints |
| `c3g-train-outputs` | `/outputs` | Hydra runs and checkpoints |
| `replica` | `/replica` | `dataset.replica_2dseg.roots=[/replica]` |
| `scannet` | `/scannet` | `dataset.scannet_2dseg.roots=[/scannet]` |

Use `+training=feature_head_sam_prompted` with the `replica_2dseg` or `scannet_2dseg` dataset group in Hydra YAML for local runs. On Modal, **`src/modal/train.py`** selects the preset:

| `--experiment` | Hydra preset |
|----------------|--------------|
| `distillation` (default) | `feature_head_sam_precomputed` |
| `prompted` | `feature_head_sam_prompted_scannet` |

### Replica

```bash
python -m src.main \
    +training=feature_head_sam_prompted \
    +dataset@_group_.replica_2dseg=replica_2dseg \
    wandb.mode=online \
    wandb.name=sam_prompted_replica \
    hydra.run.dir=/outputs/runs/sam_prompted_replica \
    dataset.replica_2dseg.roots=[/replica] \
    train.sam_checkpoint=/weights/sam_vit_h.pth \
    model.encoder.pretrained_weights=/weights/gaussian_decoder.ckpt
```

### ScanNet

```bash
python -m src.main \
    +training=feature_head_sam_prompted \
    +dataset@_group_.scannet_2dseg=scannet_2dseg \
    wandb.mode=online \
    wandb.name=sam_prompted_scannet \
    hydra.run.dir=/outputs/runs/sam_prompted_scannet \
    dataset.scannet_2dseg.roots=[/scannet] \
    train.sam_checkpoint=/weights/sam_vit_h.pth \
    model.encoder.pretrained_weights=/weights/gaussian_decoder.ckpt
```

### Common overrides

**Random-point prompts:**

```bash
train.prompt_strategy=random_point
```

**Weights & Biases on Modal** (`--wandb-mode online`):

The training container needs `WANDB_API_KEY`. Create a Modal secret once (API key from [wandb.ai/authorize](https://wandb.ai/authorize)):

```bash
modal secret create wandb WANDB_API_KEY=<your-key>
```

`wandb login` on your laptop does not apply inside Modal.

**Disable Weights & Biases:**

```bash
--wandb-mode disabled
```

**Resume** from a checkpoint on the output volume:

```bash
checkpointing.load=/outputs/runs/sam_prompted_replica/checkpoints/last.ckpt
```

### Modal CLI

**ScanNet training** (Hydra ``+training=`` only; select experiment via CLI):

```bash
modal run src/modal/train.py --wait
modal run src/modal/train.py --experiment prompted --wait
modal run src/modal/train.py::smoke --wait
modal run src/modal/train.py::smoke --experiment prompted --wait
```

**Vanilla SAM eval** (GT point prompts; no Hydra):

```bash
modal run src/modal/eval_masks.py::sam_smoke --dataset replica --wait
modal run src/modal/eval_masks.py::sam --wait
```

Shared volume paths live in `src/modal/common.py`.

Jobs spawn on Modal by default; use `--wait` to block until completion.

### Key training parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train.prompt_mode` | `prompted` | Point prompts from GT label maps |
| `train.prompted_seg_loss_weight` | `1.0` | Weight for prompted BCE + Dice loss |
| `train.prompt_strategy` | `centroid` | `centroid` or `random_point` |
| `train.min_object_pixels` | `16` | Minimum foreground pixels for a valid prompt |
| `train.sam_model_variant` | `sam_vit_h` | SAM variant |
| `train.use_lora` | `false` | LoRA on SAM decoder |
| `train.lora_rank` | `4` | LoRA rank (if enabled) |
| `model.encoder.freeze_backbone` | `true` | Freeze VGGT aggregator (Modal + prompted config) |
| `model.encoder.freeze_instill_qk` | `true` | Separate `to_q`/`to_k` (no grad, `no_grad` forward); V + `to_anotherv` train |

## Evaluation (segment-everything mode)

Evaluation uses grid prompts regardless of training `prompt_mode`. Example for Replica:

```bash
python -m src.main \
    +training=feature_head_sam_prompted \
    +dataset@_group_.replica_2dseg=replica_2dseg \
    mode=test \
    wandb.mode=online \
    wandb.name=sam_prompted_replica_eval \
    dataset.replica_2dseg.roots=[/replica] \
    train.sam_checkpoint=/weights/sam_vit_h.pth \
    checkpointing.load=/outputs/runs/sam_prompted_replica/checkpoints/last.ckpt
```

Use `scannet_2dseg` and `/scannet` for ScanNet eval runs.

## Expected outputs

### Checkpoints

```
/outputs/runs/<wandb.name>/checkpoints/
```

Defaults: top 5 checkpoints every 10,000 steps (`checkpointing.every_n_train_steps`, `checkpointing.save_top_k`).

### Evaluation visualizations

With `test.save_compare=true`, per-scene mask overlays are written under the Hydra run directory. For production 2D-seg metrics, use the mask export + scoring pipeline ([12-c3g-sam.md](12-c3g-sam.md)):

```bash
modal run src/modal/eval_masks.py::c3g --wait
modal run src/modal/get_scores.py --experiment c3gsam --wait
```

### Wandb monitoring

When `wandb.mode=online`:

- `loss/prompted_segmentation` — prompted BCE + Dice loss
- `loss/total` — combined training loss
- Reconstruction losses (MSE, LPIPS) when enabled
- Multi-view IoU during validation
