# Reading Guide

Three personas with recommended reading orders. Each entry lists the file and what to look for.

## Persona 1: Run the Model

Goal: Get C3G running for inference or evaluation as quickly as possible.

| # | File | What to Read |
|---|------|-------------|
| 1 | `README.md` | Installation, weight downloads, evaluation commands |
| 2 | `config/main.yaml` | Understand top-level config structure |
| 3 | `config/evaluation/re10k.yaml` | See what an evaluation config looks like |
| 4 | `assets/evaluation_index_re10k.json` | Understand evaluation index format |
| 5 | `src/main.py` | Entry point — how configs become a running model |
| 6 | `src/model/model_wrapper.py::test_step` | What happens during evaluation |

**Paper reference**: Section 4 (Experiments) describes evaluation protocols and datasets.

## Persona 2: Understand the Architecture

Goal: Understand how C3G predicts Gaussians from images and renders novel views.

| # | File | What to Read |
|---|------|-------------|
| 1 | Paper §3.1 | "Compact 3D Gaussian Prediction" — the core idea |
| 2 | `src/model/types.py` | `Gaussians` dataclass — the 3D representation |
| 3 | `src/dataset/types.py` | `BatchedExample` — input/output data format |
| 4 | `src/model/encoder/__init__.py` | Encoder registry and selection |
| 5 | `src/model/encoder/encoder_vggt.py` | Main encoder: VGGT backbone → Gaussian tokens → Transformer decoder |
| 6 | `src/model/encoder/common/gmae.py` | Transformer/InstillTransformer — cross-attention mechanism |
| 7 | `src/model/encoder/common/gaussian_adapter.py` | Raw outputs → valid Gaussian parameters |
| 8 | `src/model/encoder/backbone/backbone_vggt.py` | VGGT backbone wrapper |
| 9 | `src/model/decoder/decoder_splatting_cuda.py` | Differentiable Gaussian rasterization |
| 10 | `src/model/decoder/cuda_splatting.py` | Low-level render_cuda function |
| 11 | Paper §3.2 | "Feature Distillation" — VFM feature lifting |
| 12 | `src/model/load_foundation_model.py` | VFM loading dispatch |
| 13 | `src/model/model_wrapper.py::forward_foundation_model` | How VFM features are extracted |
| 14 | `src/model/model_wrapper.py::training_step` | Feature rendering loss computation |

**Paper references**:

- §3.1: Architecture overview, Gaussian token design
- §3.2: Feature distillation from VFMs
- §3.3: Training objectives (MSE + LPIPS + feature loss)
- Figure 2: Architecture diagram

## Persona 3: Extend a Module

Goal: Add a new encoder, dataset, loss, or VFM.

### Adding a New Encoder

| # | File | Action |
|---|------|--------|
| 1 | `src/model/encoder/encoder.py` | Read the `Encoder` base class |
| 2 | `src/model/encoder/encoder_vggt.py` | Study existing implementation |
| 3 | `src/model/encoder/__init__.py` | Register your encoder in `ENCODERS` dict |
| 4 | `config/model/encoder/` | Create a new YAML config |

### Adding a New Dataset

| # | File | Action |
|---|------|--------|
| 1 | `src/dataset/dataset.py` | Read `DatasetCfgCommon` base |
| 2 | `src/dataset/dataset_re10k.py` | Study the RE10K adapter pattern |
| 3 | `src/dataset/__init__.py` | Register in `DATASETS` dict and `DatasetCfgWrapper` union |
| 4 | `config/dataset/` | Create dataset YAML + view sampler config |

### Adding a New Loss

| # | File | Action |
|---|------|--------|
| 1 | `src/loss/loss.py` | Read the `Loss` base class |
| 2 | `src/loss/loss_mse.py` | Study simplest implementation |
| 3 | `src/loss/__init__.py` | Register in `LOSSES` dict and `LossCfgWrapper` union |
| 4 | `config/loss/` | Create loss YAML config |

### Adding a New Foundation Model

| # | File | Action |
|---|------|--------|
| 1 | `src/model/load_foundation_model.py` | Add loading logic for your VFM |
| 2 | `src/model/model_wrapper.py::forward_foundation_model` | Add inference branch |
| 3 | `config/training/` | Create a `feature_head_<name>.yaml` |
| 4 | CUDA rasterizer `config.h` | Update `NUM_SEMANTIC_CHANNELS` |

## Module-to-Reference Table

| Module | Paper Section | External Reference |
|--------|--------------|-------------------|
| VGGT Backbone | §3.1 | [VGGT (Wang et al., 2025)](https://arxiv.org/abs/2503.11651) |
| Gaussian Tokens | §3.1 | Novel contribution |
| DPT Head | §3.1 | [DPT (Ranftl et al., 2021)](https://arxiv.org/abs/2103.13413) |
| Gaussian Splatting | §3.1 | [3DGS (Kerbl et al., 2023)](https://arxiv.org/abs/2308.14737) |
| Feature Distillation | §3.2 | Novel contribution |
| LSeg Features | §3.2, §4.2 | [LSeg (Li et al., 2022)](https://arxiv.org/abs/2201.03546) |
| DINOv2 Features | §3.2, §4.2 | [DINOv2 (Oquab et al., 2024)](https://arxiv.org/abs/2304.07193) |
| CroCo Backbone | — | [CroCo (Weinzaepfel et al., 2023)](https://arxiv.org/abs/2210.10716) |
| NoPoSplat Architecture | — | [NoPoSplat (Ye et al., 2024)](https://arxiv.org/abs/2410.24207) |
| Pose-free Reconstruction | §3.1 | [DUSt3R (Wang et al., 2024)](https://arxiv.org/abs/2312.14132) |
| RE10K Dataset | §4.1 | [RealEstate10K (Zhou et al., 2018)](https://google.github.io/realestate10k/) |
| ScanNet Dataset | §4.2 | [ScanNet (Dai et al., 2017)](http://www.scan-net.org/) |
| Feature 3DGS Protocol | §4.2 | [Feature 3DGS (Zhou et al., 2024)](https://arxiv.org/abs/2312.03203) |
