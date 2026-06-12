# C3G-SAM Integration

**C3G + Segment Anything Model (SAM)** extends the upstream [C3G](https://arxiv.org/abs/2512.04021) codebase to distill SAM ViT-H features into compact 3D Gaussians and perform open-vocabulary 2D semantic segmentation via point-prompted SAM decoding.

For architecture details see [arch-details.md](arch-details.md). For upstream C3G training (Gaussian / VFM feature heads), see [07-training.md](07-training.md).

---

## Entry points at a glance

| Task | Local | Modal |
|------|-------|-------|
| **Precompute SAM features** | `uv run python scripts/precompute_sam_features.py …` | `modal run src/modal/precompute.py::main --dataset scannet --wait` |
| **Train — Form 1 (distillation)** | `uv run python -m src.main +training=feature_head_sam_precomputed …` | `modal run src/modal/train.py --wait` |
| **Train — Form 2 (prompted)** | `uv run python -m src.main +training=feature_head_sam_prompted_scannet …` | `modal run src/modal/train.py --experiment prompted --wait` |
| **Training smoke test** | Override `trainer.max_steps=1` in Hydra | `modal run src/modal/train.py::smoke --wait` |
| **Export masks (vanilla SAM)** | `scripts/run_local.sh eval-sam` | `modal run src/modal/eval_masks.py::sam --wait` |
| **Export masks (C3G-SAM + ablations)** | `python -m src.evaluation.mask_export +evaluation=c3g_sam_distill checkpointing.load=…` | `modal run src/modal/eval_masks.py::c3g --wait` |
| **Score exported masks** | `python -m src.evaluation.score_masks --experiment c3gsam` | `modal run src/modal/get_scores.py --experiment c3gsam --wait` |
| **Seg viz / loss plots** | `python -m src.tools.seg_viz` · `python -m src.tools.loss_plots` | `modal run src/tools/seg_viz.py --wait` |
| **Download datasets** | `python -m src.dataset.download_replica …` | `modal run src/dataset/download_replica.py` |

**Shell wrappers:** `scripts/run_local.sh` and `scripts/run_modal.sh` cover precompute, train, eval, ablations, scoring, and viz.

**Modal via main.py:** `python -m src.main --modal train --experiment distillation --wait` (also `precompute`, `eval`, `score`, `viz`).

Modal jobs detach by default; pass `--wait` to block until completion. Shared volume names and mount paths live in [`src/modal/common.py`](../src/modal/common.py). Local dataset and prediction defaults live in [`src/evaluation/eval_common.py`](../src/evaluation/eval_common.py).

---

## Training formulations

### Form 1 — Feature distillation (precomputed SAM)

Distills rendered Gaussian features to match **offline SAM ViT-H encoder outputs** (`{frame_id}_sam.pt`, shape 256×64×64). SAM is **not** loaded during training.

| | |
|---|---|
| Lightning wrapper | `DistillationModelWrapper` |
| Hydra preset (ScanNet / Modal) | `+training=feature_head_sam_precomputed` |
| Modal CLI | `modal run src/modal/train.py` (default `--experiment distillation`) |
| Dataset config | `scannet_distill` (Modal: `/scannet` + `/precompute_sam_features/scannet`) |
| Primary losses | Cosine similarity + feature magnitude MSE |
| Checkpoint selection | Best by training step (`every_n_train_steps: 50`, `save_top_k: 20`) |

**Local example (after precomputing features):**

```bash
uv run python -m src.main +training=feature_head_sam_precomputed \
    dataset.scannet_distill.roots=[./datasets/scannet] \
    dataset.scannet_distill.sam_features_root=./datasets/scannet \
    model.encoder.pretrained_weights=./pretrained_weights/gaussian_decoder.ckpt \
    wandb.mode=disabled
```

See [distillation_training.md](distillation_training.md) for precompute options and parameter tables.

### Form 2 — Prompted segmentation (live SAM)

End-to-end training with **point prompts from GT labels** and SAM mask-decoder supervision.

| | |
|---|---|
| Lightning wrapper | `ModelWrapper` |
| Hydra preset (ScanNet / Modal) | `+training=feature_head_sam_prompted_scannet` |
| Modal CLI | `modal run src/modal/train.py --experiment prompted --wait` |
| Dataset config | `scannet_2dseg` (Modal: `/scannet`; optional precomputed features for debug) |
| Primary losses | Prompted BCE + Dice + RGB MSE + LPIPS |
| Checkpoint selection | **Best `val/loss`** (not IoU or train step) |

**Local example (Replica):**

```bash
uv run python -m src.main +training=feature_head_sam_prompted \
    +dataset@_group_.replica_2dseg=replica_2dseg \
    dataset.replica_2dseg.roots=[./datasets/replica] \
    train.sam_checkpoint=./pretrained_weights/sam_vit_h.pth \
    wandb.mode=online wandb.name=sam_prompted_replica
```

See [prompted_training.md](prompted_training.md) (local) and [prompted_training_modal.md](prompted_training_modal.md) (Modal volumes and dataset prep).

---

## Modal volumes

All GPU workflows use persistent [Modal](https://modal.com/) volumes. Upload once, then reference by mount path inside Hydra configs.

| Volume name | Mount path | Contents |
|-------------|------------|----------|
| `c3g-weights` | `/weights` | Base weights and trained checkpoints (see table below) |
| `replica` | `/replica` | Prepared Replica 2D-seg scenes (8 scenes) |
| `scannet` | `/scannet` | Prepared ScanNet scenes (`scene0000_00` … `scene0806_00`) |
| `precompute_sam_features` | `/precompute_sam_features` | `{dataset}/{scene}/{frame_id}_sam.pt` |
| `c3g-train-outputs` | `/outputs` | Training runs → `/outputs/runs/<wandb.name>/checkpoints/` |
| `vanilla-sam-outputs` | `/vanilla-sam-outputs` | Vanilla SAM mask exports |
| `c3g-sam-eval-outputs` | `/c3g-sam-eval-outputs` | Main C3G-SAM mask exports |
| `c3g-sam-dft-eval-outputs` | `/c3g-sam-dft-eval-outputs` | DFT ablation mask exports |
| `c3g-sam-nomaghead-eval-outputs` | `/c3g-sam-nomaghead-eval-outputs` | No-magnitude-head ablation exports |
| `c3g-sam-ema-nomag-eval-outputs` | `/c3g-sam-ema-nomag-eval-outputs` | EMA-without-mag-head ablation exports |

**Upload base weights (once):**

```bash
modal volume put c3g-weights ./pretrained_weights/sam_vit_h.pth sam_vit_h.pth
modal volume put c3g-weights ./pretrained_weights/gaussian_decoder.ckpt gaussian_decoder.ckpt
```

**W&B on Modal:** create secret `modal secret create wandb WANDB_API_KEY=<key>`. Smoke configs set `wandb.mode=disabled`.

---

## Checkpoints on `c3g-weights`

Files below are stored at the **root** of the `c3g-weights` volume (`/weights/<filename>` in containers).

| File on volume | Eval alias | Description |
|----------------|--------------|-------------|
| `sam_vit_h.pth` | `sam` (baseline) | Frozen SAM ViT-H (mask decoder at eval) |
| `gaussian_decoder.ckpt` | — | C3G encoder init for all SAM training |
| `distillation-base.ckpt` | `c3gsam` | **Main model**: EMA norm tracking + magnitude head + standard learnable tokens |
| `distillation-diff_learnable_tokens.ckpt` | `c3gsam_ema-mag-uproj` | DFT ablation: `different_learnable_tokens=true` + up-projection |
| `c3gsam_ema.ckpt` | `c3gsam_ema` | No EMA + no magnitude head |
| `ema-nomag.ckpt` | `c3gsam_noema-nomag` | EMA norm tracking, no magnitude head |

Training outputs (new runs) land on **`c3g-train-outputs`**, not `c3g-weights`. Copy a finished checkpoint to `c3g-weights` before running eval:

```bash
modal volume get c3g-train-outputs runs/sam_distill_scannet/checkpoints/epoch=…ckpt ./local.ckpt
modal volume put c3g-weights ./local.ckpt distillation-base.ckpt
```

---

## Ablations and evaluation mapping

Each eval variant uses the same export pipeline ([`modal_eval_masks.py`](../src/modal/eval_masks.py), [`mask_export.py`](../src/evaluation/mask_export.py)) with a different checkpoint and output volume.

| Method | Checkpoint (`c3g-weights`) | Mask export volume | Scoring `--experiment` | Modal export entrypoint |
|--------|---------------------------|--------------------|------------------------|-------------------------|
| Vanilla SAM | `sam_vit_h.pth` | `vanilla-sam-outputs` | `sam` | `modal_eval_masks.py::sam` |
| C3G-SAM (main) | `distillation-base.ckpt` | `c3g-sam-eval-outputs` | `c3gsam` | `modal_eval_masks.py::c3g` |
| C3G-SAM DFT | `distillation-diff_learnable_tokens.ckpt` | `c3g-sam-dft-eval-outputs` | `c3gsam_ema-mag-uproj` | `modal_eval_masks.py::c3gsam_ema-mag-uproj` |
| C3G-SAM no mag head | `c3gsam_ema.ckpt` | `c3g-sam-nomaghead-eval-outputs` | `c3gsam_ema` | `modal_eval_masks.py::c3gsam_ema` |
| C3G-SAM EMA, no mag | `ema-nomag.ckpt` | `c3g-sam-ema-nomag-eval-outputs` | `c3gsam_noema-nomag` | `modal_eval_masks.py::c3gsam_noema-nomag` |

**Full eval workflow (Modal):**

```bash
# 1. Export per-class masks (+ logits) for each method
modal run src/modal/eval_masks.py::sam --wait
modal run src/modal/eval_masks.py::c3g --wait
modal run src/modal/eval_masks.py::c3gsam_ema-mag-uproj --wait
modal run src/modal/eval_masks.py::c3gsam_ema --wait
modal run src/modal/eval_masks.py::c3gsam_noema-nomag --wait

# 2. Score against GT (Replica all scenes + ScanNet test split)
modal run src/modal/get_scores.py --experiment sam --wait
modal run src/modal/get_scores.py --experiment c3gsam --wait
modal run src/modal/get_scores.py::c3gsam_ema-mag-uproj --wait
modal run src/modal/get_scores.py::c3gsam_ema --wait
modal run src/modal/get_scores.py::c3gsam_noema-nomag --wait
```

Scoring writes `scores.json` and `scores_by_scene.csv` on the corresponding eval output volume. Metrics: global pixel IoU, boundary IoU, and warp mIoU (see [08-evaluation.md](08-evaluation.md#c3g-sam-mask-evaluation)).

Pre-generated figures and tables live under [`c3gsam_results/`](../c3gsam_results/). Regenerate with `modal run src/tools/seg_viz.py` and `python -m src.tools.loss_plots`.

---

## Dataset layout

Flat per-scene layout (Replica and ScanNet):

```
<root>/
├── <scene_id>/
│   ├── 00000_x.jpg          # RGB
│   ├── 00000_y.png          # semantic labels (uint16)
│   ├── 00000_cam.npz        # camera_pose (4×4), camera_intrinsics (3×3)
│   ├── 00000_sam.pt         # precomputed SAM features (optional / required for distillation)
│   └── ...
```

**ScanNet splits** (`assets/scannet_2dseg_scene_splits.json`): 775 train, 8 val, 24 test. Eval and scoring use Replica (all 8 scenes) + ScanNet **test** only.

---

## CUDA rasterizer note

SAM uses **256**-dim Gaussian features. Before building the feature-detach rasterizer, set `NUM_SEMANTIC_CHANNELS` to **256** in:

`submodules/diff_gaussian_rasterization_w_feature_detach/cuda_rasterizer/config.h`

Then reinstall: `uv pip install -e submodules/diff_gaussian_rasterization_w_feature_detach`

---

## Related documents

| Document | Topic |
|----------|-------|
| [distillation_training.md](distillation_training.md) | Form 1 — precompute + local distillation |
| [prompted_training.md](prompted_training.md) | Form 2 — local prompted training |
| [prompted_training_modal.md](prompted_training_modal.md) | Modal dataset prep and volume layout |
| [arch-details.md](arch-details.md) | Tensor shapes, wrappers, config tables |
| [05-datasets.md](05-datasets.md) | Dataset adapter reference |
