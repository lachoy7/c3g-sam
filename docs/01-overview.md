# Repository Overview

## What is C3G?

C3G (**C**ompact **3**D representations with 2K **G**aussians) is a feed-forward framework that takes unposed images as input and predicts approximately 2048 3D Gaussians. These Gaussians are rendered via differentiable splatting to produce novel views. The model also supports feature distillation from Vision Foundation Models (VFMs) for 3D scene understanding tasks like semantic segmentation.

**Paper**: [C3G: Learning Compact 3D Representations with 2K Gaussians](https://arxiv.org/abs/2512.04021) (CVPR 2026)

**Project Page**: [https://cvlab-kaist.github.io/C3G](https://cvlab-kaist.github.io/C3G)

## Key Contributions

1. A feed-forward model that predicts only ~2K Gaussians allocated in meaningful regions, enabling compact yet expressive 3D scene representations.
2. Pose-free reconstruction — no ground-truth camera poses required at inference.
3. Feature distillation from VFMs (LSeg, DINOv2, DINOv3, VGGT) into Gaussian features for open-vocabulary 3D scene understanding.
4. State-of-the-art novel view synthesis on RealEstate10K and competitive 3D semantic segmentation on ScanNet/Replica.

## C3G-SAM extension (this fork)

This repository adds **SAM ViT-H** integration on top of upstream C3G:

1. **Form 1 — Feature distillation**: match precomputed SAM encoder features via Gaussian splatting (`DistillationModelWrapper`).
2. **Form 2 — Prompted segmentation**: train with point prompts from GT labels and SAM mask-decoder loss (`ModelWrapper`).

See **[12-c3g-sam.md](12-c3g-sam.md)** for training entry points (local and Modal), Modal volumes, checkpoint names, and ablation eval mapping.

## Tech Stack

| Component | Version / Tool |
|-----------|---------------|
| Language | Python 3.11 |
| Deep Learning | PyTorch 2.5.1 |
| GPU | CUDA 12.4 |
| Training Framework | PyTorch Lightning |
| Configuration | Hydra (OmegaConf) |
| Gaussian Rasterization | gsplat 1.5.3 + custom CUDA rasterizer |
| Experiment Tracking | Weights & Biases |
| Type Checking | jaxtyping + beartype |

## Directory Structure

```
C3G/
├── config/                     # Hydra configuration files
│   ├── main.yaml               # Root config
│   ├── dataset/                # Dataset configs (re10k, scannet, replica, etc.)
│   │   └── view_sampler/       # View sampling strategies
│   ├── evaluation/             # Evaluation presets (re10k, re10k_multiview, scannet)
│   ├── loss/                   # Loss function configs
│   ├── model/                  # Model configs (encoder, decoder)
│   └── training/               # Training presets (gaussian_head, feature_head_*)
├── src/                        # Source code
│   ├── main.py                 # Entry point (Hydra-decorated train function)
│   ├── config.py               # Typed config loading (RootCfg dataclass)
│   ├── global_cfg.py           # Global config singleton
│   ├── dataset/                # Data loading, datasets, view samplers, shims
│   ├── evaluation/             # Metrics, pose evaluator, index generator
│   ├── geometry/               # Camera math, epipolar lines, projection
│   ├── loss/                   # Loss implementations (MSE, LPIPS, SSIM, etc.)
│   ├── misc/                   # Utilities (logging, image I/O, benchmarking)
│   ├── model/                  # Model code
│   │   ├── encoder/            # Encoder variants (NoPoSplat, VGGT)
│   │   │   ├── backbone/       # Backbones (CroCo, VGGT, DINOv2, ResNet)
│   │   │   └── common/         # Shared components (GaussianAdapter, Transformer)
│   │   ├── decoder/            # Gaussian splatting decoder
│   │   ├── clip/               # CLIP model for text-feature matching
│   │   ├── lseg/               # LSeg feature extractor
│   │   ├── distiller/          # Knowledge distillation utilities
│   │   ├── model_wrapper.py    # LightningModule (Form 2 prompted training)
│   │   ├── distillation_wrapper.py  # Form 1 distillation Lightning module
│   │   ├── sam/                # SAM ViT-H loader and forward pass
│   │   ├── sam_decoder.py      # SAM mask decoder wrapper
│   │   ├── prompt_sampler.py   # Point prompts from label maps
│   │   └── lora.py             # Optional LoRA on SAM decoder
│   ├── modal/                  # Modal apps (train, precompute, eval, scoring)
│   ├── tools/                  # Local viz & plotting (seg_viz, loss_plots, data_examples)
│   ├── evaluation/             # mask_export, score_masks, eval_common
│   └── visualization/          # Rendering, annotation, camera trajectories
├── assets/                     # Evaluation indices, ScanNet splits, teaser image
├── c3gsam_results/             # Pre-generated loss curves and seg comparisons
├── docs/                       # Documentation (see docs/README.md)
├── pretrained_weights/         # Downloaded model weights (not in git)
└── pyproject.toml              # Project metadata (uv lockfile: uv.lock)
```

## Related Projects

| Project | Role in C3G |
|---------|-------------|
| [VGGT](https://github.com/facebookresearch/vggt) | Backbone encoder (VGGT-1B) and VFM for feature distillation |
| [NoPoSplat](https://github.com/cvg/NoPoSplat) | Architectural foundation for pose-free Gaussian prediction |
| [CroCo](https://github.com/naver/croco) | Cross-view completion backbone |
| [DUSt3R](https://github.com/naver/dust3r) | Inspiration for pose-free 3D from image pairs |
| [pixelSplat](https://github.com/dcharatan/pixelsplat) | RE10K data preprocessing pipeline |
| [MVSplat](https://github.com/donydchen/mvsplat) | RE10K data preprocessing pipeline |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | Differentiable Gaussian splatting library |
| [LSeg](https://github.com/isl-org/lang-seg) | Language-driven semantic segmentation features |
| [Feature 3DGS](https://github.com/ShijieZhou-UCLA/feature-3dgs) | Replica evaluation protocol |
