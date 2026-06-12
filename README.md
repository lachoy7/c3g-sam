# C3G-SAM

Compact 3D Gaussians with SAM ViT-H feature distillation and prompted segmentation

> **C3G-SAM** extends [C3G: Learning Compact 3D Representations with 2K Gaussians](https://arxiv.org/abs/2512.04021) (CVPR 2026) with [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) integration. Multi-view context images pass through a **VGGT encoder** and an **Instill Transformer** with separate **Geometry** and **Feature** streams to produce 2048 3D Gaussians. Rendered Gaussian features are either matched to precomputed SAM encoder outputs (distillation) or decoded into segmentation masks via a frozen **SAM mask decoder** (prompted training).

**Every major workflow runs locally or on [Modal](https://modal.com)** — precompute, training (distillation and prompted), mask export, scoring, ablations, and figure generation. Modal apps live under [`src/modal/`](src/modal/); use [`scripts/run_modal.sh`](scripts/run_modal.sh) or `python -m src.main --modal …` without changing Hydra configs. See [docs/12-c3g-sam.md](docs/12-c3g-sam.md) for volumes and checkpoints.

This repository is a fork of the upstream [C3G codebase](https://github.com/cvlab-kaist/C3G). We gratefully acknowledge the original authors:

[Honggyu An](https://hg010303.github.io/) · [Jaewoo Jung](https://crepejung00.github.io/) · Mungyeom Kim · [Chaehyun Kim](https://kchyun.github.io/) · [Minkyeong Jeon](https://sites.google.com/view/minjeon/home) · [Jisang Han](https://onground.github.io/) · Kazumi Fukuda · Takuya Narihira · Hyuna Ko · Junsu Kim · [Sunghwan Hong](https://sunghwanhong.github.io/) · [Yuki Mitsfuji](https://www.yukimitsufuji.com/) · [Seungryong Kim](https://cvlab.kaist.ac.kr/members/faculty)

[C3G Paper](https://arxiv.org/abs/2512.04021) · [C3G Project Page](https://cvlab-kaist.github.io/C3G) · Full documentation

---

## Architecture

The diagram above shows the full C3G-SAM pipeline:


| Component                         | Role                                                                                                      |
| --------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Context images**                | Multi-view RGB input (224×224 at train time)                                                              |
| **Gaussian tokens (×2048)**       | Learnable slots that become 3D Gaussians                                                                  |
| **VGGT encoder**                  | Patch tokens for geometry; drives the Geometry stream                                                     |
| **SAM encoder** *(purple)*        | ViT-H image encoder — produces 256-d patch features for the Feature stream                                |
| **Instill Transformer**           | Dual-stream cross-attention: **Geometry stream** (3D layout + RGB) and **Feature stream** (SAM semantics) |
| **Gaussian decoder + rasterizer** | Differentiable splatting → rendered RGB and feature maps                                                  |
| **SAM decoder** *(purple)*        | Frozen mask decoder — turns rendered features + point prompts into masks                                  |
| **Target mask**                   | Per-class segmentation output at the target view                                                          |


Shared Q/K attention couples the two Instill streams so each Gaussian slot aligns with the corresponding image region's SAM embedding. See [docs/arch-details.md](docs/arch-details.md) for tensor shapes and config tables.

---

## Training formulations

C3G-SAM supports two training modes. They differ in **what is frozen**, **what supervision is used**, and **whether SAM runs live** during training.

### Form 1 — Feature distillation

Offline SAM ViT-H encoder features (`{frame_id}_sam.pt`, 256×64×64) are precomputed once. At train time SAM is **not loaded** — the model learns to reproduce encoder features via Gaussian splatting.


|                   |                                                                                                                                                                                                        |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **What's frozen** | Only the **V projection head** (`to_anotherv`) in the Feature stream of the Instill Transformer. Q/K, the Geometry stream, VGGT backbone, Gaussian decoder, and feature output heads remain trainable. |
| **Losses**        | Cosine similarity + feature magnitude MSE between rendered Gaussian features and precomputed SAM features                                                                                              |
| **Hydra preset**  | `+training=feature_head_sam_precomputed`                                                                                                                                                               |
| **Modal**         | `modal run src/modal/train.py --wait`                                                                                                                                                                  |


```bash
# Precompute (required once)
uv run python scripts/precompute_sam_features.py \
    --dataset-root ./datasets/scannet --dataset scannet \
    --sam-checkpoint ./pretrained_weights/sam_vit_h.pth

# Train
uv run python -m src.main +training=feature_head_sam_precomputed \
    dataset.scannet_distill.roots=[./datasets/scannet] \
    dataset.scannet_distill.sam_features_root=./datasets/scannet \
    model.encoder.pretrained_weights=./pretrained_weights/gaussian_decoder.ckpt
```

### Form 2 — Prompted segmentation

The full C3G pipeline is trained end-to-end with **point prompts** sampled from GT semantic labels and supervision through the frozen SAM mask decoder.


|                          |                                                                                                                                                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What's frozen**        | **SAM encoder** and **SAM mask decoder** only (purple blocks in the diagram). The entire C3G pipeline — VGGT encoder, Instill Transformer (both streams), Gaussian decoder, and rasterizer — is **unfrozen**. |
| **Losses**               | Prompted BCE + Dice on segmentation masks, plus RGB MSE and LPIPS reconstruction                                                                                                                              |
| **Hydra preset**         | `+training=feature_head_sam_prompted_scannet` (ScanNet) or `+training=feature_head_sam_prompted` + `replica_2dseg` (Replica)                                                                                  |
| **Modal**                | `modal run src/modal/train.py --experiment prompted --wait`                                                                                                                                                   |
| **Checkpoint selection** | Best `**val/loss`**, not IoU                                                                                                                                                                                  |


```bash
uv run python -m src.main +training=feature_head_sam_prompted \
    +dataset@_group_.replica_2dseg=replica_2dseg \
    dataset.replica_2dseg.roots=[./datasets/replica] \
    train.sam_checkpoint=./pretrained_weights/sam_vit_h.pth \
    model.encoder.pretrained_weights=./pretrained_weights/gaussian_decoder.ckpt
```


|                   | Form 1 — Distillation                          | Form 2 — Prompted              |
| ----------------- | ---------------------------------------------- | ------------------------------ |
| SAM at train time | No (precomputed `.pt` files)                   | Yes (frozen encoder + decoder) |
| C3G pipeline      | Mostly trainable; Feature-stream V proj frozen | Fully trainable                |
| Supervision       | Dense encoder features                         | Segmentation masks via prompts |
| Primary losses    | Cosine + magnitude MSE                         | BCE + Dice + RGB MSE + LPIPS   |


Detailed guides: [distillation_training.md](docs/distillation_training.md) · [prompted_training.md](docs/prompted_training.md) · [prompted_training_modal.md](docs/prompted_training_modal.md)

---

## Installation

Python 3.11+, PyTorch 2.5.1, CUDA 12.4.

```bash
uv sync --frozen
source .venv/bin/activate
```

Install [uv](https://docs.astral.sh/uv/) if needed: `curl -LsSf https://astral.sh/uv/install.sh | sh`

Before building the feature rasterizer, set `NUM_SEMANTIC_CHANNELS=256` in `submodules/diff_gaussian_rasterization_w_feature_detach/cuda_rasterizer/config.h`, then:

```bash
uv pip install -e submodules/diff_gaussian_rasterization_w_feature_detach
```

### Weights

```bash
mkdir -p pretrained_weights
wget https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt -O ./pretrained_weights/model.pt
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O ./pretrained_weights/sam_vit_h.pth
# C3G Gaussian decoder init → pretrained_weights/gaussian_decoder.ckpt
# (from https://huggingface.co/honggyuAn/C3G or your own checkpoint)
```

### Datasets

Replica and ScanNet use a flat per-scene layout (`{frame_id}_x.jpg`, `{frame_id}_y.png`, `{frame_id}_cam.npz`):

```bash
python -m src.dataset.download_replica --source /path/to/raw/replica --out-dir ./datasets/replica
python -m src.dataset.download_scannet --out-dir ./datasets/scannet --accept-tos
```

ScanNet splits: 775 train / 8 val / 24 test (`assets/scannet_2dseg_scene_splits.json`).

---

## Modal (cloud GPUs)

This repo is built for **local development and Modal production runs** with the same Hydra presets and eval pipeline.

1. **Install Modal** (once): `pip install modal && modal setup`
2. **Populate volumes** — datasets (`replica`, `scannet`), weights (`c3g-weights`), and optional precomputed SAM features. Details in [docs/12-c3g-sam.md](docs/12-c3g-sam.md).
3. **Run jobs** — any of:

```bash
# Shell wrappers (recommended)
scripts/run_modal.sh train-distill --wait
scripts/run_modal.sh train-prompted --wait
scripts/run_modal.sh precompute --dataset scannet --wait
scripts/run_modal.sh eval-c3gsam --wait
scripts/run_modal.sh eval-ablation c3gsam_ema-mag-uproj --wait
scripts/run_modal.sh score --experiment c3gsam --wait

# Or via main.py
python -m src.main --modal train --experiment distillation --wait
python -m src.main --modal score --experiment c3gsam --wait

# Or direct Modal entrypoints
modal run src/modal/train.py --wait
modal run src/modal/eval_masks.py::c3g --wait
modal run src/modal/get_scores.py --experiment c3gsam --wait
```

Training checkpoints are written to the **`c3g-train-outputs`** volume at `/outputs/runs/<wandb.name>/checkpoints/`. Modal jobs detach by default; pass `--wait` (or set `C3G_MODAL_WAIT=1` in the shell scripts) to block until completion.

For a full local vs Modal command matrix, see **Entry points** below and [docs/prompted_training_modal.md](docs/prompted_training_modal.md).

---

## Entry points

Local and Modal commands are paired below. Prefer **`scripts/run_local.sh`** / **`scripts/run_modal.sh`** for the full train → eval → score workflow.
| Task                        | Local                                            | Modal                                                   |
| --------------------------- | ------------------------------------------------ | ------------------------------------------------------- |
| Precompute SAM features     | `scripts/run_local.sh precompute`                | `scripts/run_modal.sh precompute --wait`                |
| Train Form 1 (distillation) | `scripts/run_local.sh train-distill`             | `scripts/run_modal.sh train-distill --wait`             |
| Train Form 2 (prompted)     | `scripts/run_local.sh train-prompted`            | `scripts/run_modal.sh train-prompted --wait`            |
| Smoke test (1 step)         | add `--smoke` to train commands above            | add `--smoke` to train commands above                   |
| Export masks (SAM / C3G)    | `scripts/run_local.sh eval-sam` / `eval-c3gsam`  | `scripts/run_modal.sh eval-sam` / `eval-c3gsam --wait`  |
| Score masks                 | `scripts/run_local.sh score --experiment c3gsam` | `scripts/run_modal.sh score --experiment c3gsam --wait` |
| Seg viz / loss plots        | `scripts/run_local.sh viz` / `loss-plots`        | `scripts/run_modal.sh viz --wait`                       |


Modal via `main.py`: `python -m src.main --modal train --experiment distillation --wait` · Quick start: [Modal (cloud GPUs)](#modal-cloud-gpus) above.

Full volume and ablation reference: [docs/12-c3g-sam.md](docs/12-c3g-sam.md).

---

## Evaluation & ablations

Upload trained checkpoints to Modal volume `**c3g-weights`** before eval.


| Method                      | Checkpoint on `c3g-weights`               | Export volume                    | Score                               |
| --------------------------- | ----------------------------------------- | -------------------------------- | ----------------------------------- |
| Vanilla SAM                 | `sam_vit_h.pth`                           | `vanilla-sam-outputs`            | `--experiment sam`                  |
| C3G-SAM (main)              | `distillation-base.ckpt`                  | `c3g-sam-eval-outputs`           | `--experiment c3gsam`               |
| C3G-SAM EMA + mag + up-proj | `distillation-diff_learnable_tokens.ckpt` | `c3g-sam-dft-eval-outputs`       | `--experiment c3gsam_ema-mag-uproj` |
| C3G-SAM EMA (no mag head)   | `c3gsam-nomaghead.ckpt`                   | `c3g-sam-nomaghead-eval-outputs` | `--experiment c3gsam_ema`           |
| C3G-SAM no EMA, no mag      | `ema-nomag.ckpt`                          | `c3g-sam-ema-nomag-eval-outputs` | `--experiment c3gsam_noema-nomag`   |


Metrics: global pixel IoU, boundary IoU, and warp mIoU on Replica (8 scenes) + ScanNet test (24 scenes). Pre-generated figures: `[c3gsam_results/](c3gsam_results/)`.

---

## Documentation


| Document                                     | Description                                         |
| -------------------------------------------- | --------------------------------------------------- |
| [docs/12-c3g-sam.md](docs/12-c3g-sam.md)     | Entry points, Modal volumes, checkpoints, ablations |
| [docs/arch-details.md](docs/arch-details.md) | Shape-annotated architecture reference              |
| [docs/README.md](docs/README.md)             | Full documentation index                            |


For upstream C3G training (Gaussian decoder, LSeg/DINO feature heads, RE10K eval), see the [original C3G repository](https://github.com/cvlab-kaist/C3G).

---

## Citation

If you use the upstream C3G framework, please cite:

```bibtex
@article{an2025c3g,
  title={C3G: Learning Compact 3D Representations with 2K Gaussians},
  author={An, Honggyu and Jung, Jaewoo and Kim, Mungyeom and Hong, Sunghwan and Kim, Chaehyun and Fukuda, Kazumi and Jeon, Minkyeong and Han, Jisang and Narihira, Takuya and Ko, Hyuna and others},
  journal={arXiv preprint arXiv:2512.04021},
  year={2025}
}
```

## Acknowledgements

This work builds on [C3G](https://arxiv.org/abs/2512.04021) by An et al. and integrates [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) by Kirillov et al. We also thank the authors of [VGGT](https://github.com/facebookresearch/vggt) and [NoPoSplat](https://github.com/cvg/NoPoSplat), whose code forms the foundation of the upstream C3G project.