# C3G Documentation

Comprehensive documentation for the **C3G: Learning Compact 3D Representations with 2K Gaussians** codebase (CVPR 2026).

## Table of Contents

| # | Document | Description |
|---|----------|-------------|
| 1 | [Overview](01-overview.md) | What C3G is, key contributions, directory structure, tech stack |
| 2 | [Getting Started](02-getting-started.md) | Installation, pretrained weights, dataset preparation, commands |
| 3 | [Architecture](03-architecture.md) | End-to-end data flow, types, encoder/decoder design |
| 4 | [Configuration](04-configuration.md) | Hydra setup, config composition, dataclass mapping |
| 5 | [Datasets & View Sampling](05-datasets.md) | Dataset adapters, view samplers, evaluation indices |
| 6 | [Models & Foundation Models](06-models.md) | Encoders, decoder, Gaussians, DPT heads, VFM loading |
| 7 | [Training & Losses](07-training.md) | Training step, loss functions, feature distillation loss |
| 8 | [Evaluation & Metrics](08-evaluation.md) | Test step, pose alignment, PSNR/SSIM/LPIPS, mIoU |
| 9 | [Reading Guide](09-reading-guide.md) | Persona-based file reading orders with paper references |
| 10 | [Code Flow Walkthroughs](10-walkthroughs.md) | Step-by-step traces of training and test steps |
| 11 | [Glossary](11-glossary.md) | Definitions of key terms and acronyms |
| 12 | [C3G-SAM Integration](12-c3g-sam.md) | **Entry points**, Modal volumes, checkpoints, ablations, eval |

### C3G-SAM guides

| Document | Description |
|----------|-------------|
| [C3G-SAM Integration](12-c3g-sam.md) | Canonical hub — local vs Modal training, volumes, ablation mapping |
| [Distillation training](distillation_training.md) | Form 1 — precomputed SAM features (local) |
| [Prompted training](prompted_training.md) | Form 2 — point-prompted segmentation (local) |
| [Prompted training (Modal)](prompted_training_modal.md) | Dataset volumes, ScanNet splits, Modal training |
| [Architecture details](arch-details.md) | Shape-annotated C3G-SAM pipeline reference |

## Quick Links

- [Paper (arXiv)](https://arxiv.org/abs/2512.04021)
- [Project Page](https://cvlab-kaist.github.io/C3G)
- [Pretrained Weights (HuggingFace)](https://huggingface.co/honggyuAn/C3G/tree/main)
- [VGGT](https://github.com/facebookresearch/vggt)
- [NoPoSplat](https://github.com/cvg/NoPoSplat)
