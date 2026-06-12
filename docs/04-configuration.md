# Configuration System

## Hydra Setup

C3G uses [Hydra](https://hydra.cc/) for hierarchical configuration management. The entry point is decorated with:

```python
@hydra.main(version_base=None, config_path="../config", config_name="main")
def train(cfg_dict: DictConfig):
    ...
```

Hydra resolves `config/main.yaml` as the root, composes in defaults, and applies CLI overrides. The output directory is set per-run:

```yaml
hydra:
  run:
    dir: outputs/exp_${wandb.name}/${now:%Y-%m-%d_%H-%M-%S}
```

## Main Config Structure (`config/main.yaml`)

```yaml
defaults:
  - model/encoder: noposplat       # encoder architecture
  - model/decoder: splatting_cuda  # decoder (only option)
  - loss: [mse]                    # loss functions

wandb:           # W&B project/entity/name/mode
mode: train      # "train" or "test"
data_loader:     # batch_size, num_workers per stage
optimizer:       # lr, warm_up_steps, backbone_lr_multiplier
checkpointing:   # load path, save frequency
train:           # TrainCfg fields
test:            # TestCfg fields
seed: 111123
trainer:         # max_steps, val_check_interval, gradient_clip_val
```

## Training Configs (`config/training/`)

Training presets override the main config using `+training=<name>`:

| Config | Purpose | Key Differences |
|--------|---------|-----------------|
| `gaussian_head` | 2-view Gaussian decoder | encoder.name=vggt, num_gaussians=2048, losses=[mse,lpips] |
| `gaussian_head_multiview` | Multi-view Gaussian decoder | num_context_views=24, random_select_context_view=true |
| `feature_head_lseg` | LSeg feature distillation | gaussian_feature_dim=512, reproj_model=lseg, feature_rendering_loss=0.01 |
| `feature_head_dinov2_B` | DINOv2-B feature distillation | gaussian_feature_dim=768, reproj_model=dinov2_B |
| `feature_head_dinov2_L` | DINOv2-L feature distillation | gaussian_feature_dim=1024, reproj_model=dinov2_L |
| `feature_head_dinov3_L` | DINOv3-L feature distillation | gaussian_feature_dim=1024, reproj_model=dinov3_L |
| `feature_head_vggt` | VGGT feature distillation | gaussian_feature_dim=128, reproj_model=vggt_tracking |
| `feature_head_lseg_multiview` | Multi-view LSeg distillation | Combines multiview + LSeg |

Each training config uses `# @package _global_` to override at the root level and specifies its own defaults list:

```yaml
# @package _global_
defaults:
  - /dataset@_group_.re10k: re10k
  - override /model/encoder: noposplat
  - override /model/encoder/backbone: croco
  - override /loss: [mse, lpips]
```

## Evaluation Configs (`config/evaluation/`)

| Config | Dataset | Key Settings |
|--------|---------|--------------|
| `re10k` | RE10K (2-view) | num_context_views=24, save_top_k=5 |
| `re10k_multiview` | RE10K (multi-view) | Uses `re10k_eval` dataset variant |
| `scannet` | ScanNet | gaussian_feature_dim=512, reproj_model=lseg, attention_instill=true |

## Config-to-Dataclass Mapping

The raw `DictConfig` is converted to typed Python dataclasses in `src/config.py`:

```python
cfg = load_typed_root_config(cfg_dict)  # → RootCfg
```

| Config Key | Dataclass | Location |
|-----------|-----------|----------|
| (root) | `RootCfg` | `src/config.py` |
| `model` | `ModelCfg` | `src/config.py` |
| `model.encoder` | `EncoderNoPoSplatCfg` or `EncoderVGGTCfg` | `src/model/encoder/` |
| `model.decoder` | `DecoderSplattingCUDACfg` | `src/model/decoder/` |
| `optimizer` | `OptimizerCfg` | `src/model/model_wrapper.py` |
| `train` | `TrainCfg` | `src/model/model_wrapper.py` |
| `test` | `TestCfg` | `src/model/model_wrapper.py` |
| `checkpointing` | `CheckpointingCfg` | `src/config.py` |
| `trainer` | `TrainerCfg` | `src/config.py` |
| `data_loader` | `DataLoaderCfg` | `src/dataset/data_module.py` |
| `loss` | `list[LossCfgWrapper]` | `src/loss/__init__.py` |
| `dataset` | `list[DatasetCfgWrapper]` | `src/dataset/__init__.py` |

The conversion uses [dacite](https://github.com/konradhalas/dacite) with custom type hooks for `Path`, `list[LossCfgWrapper]`, and `list[DatasetCfgWrapper]`.

## Foundation Model Selection (`train.reproj_model`)

The `reproj_model` field in `TrainCfg` controls which VFM is loaded:

| `reproj_model` value | Foundation Model | Feature Dim | Source |
|---------------------|-----------------|-------------|--------|
| `"none"` | None | 0 | — |
| `"vggt"` | VGGT-1B | 2048 | HuggingFace hub |
| `"vggt_tracking"` | VGGT-1B (tracking features) | 2048 | HuggingFace hub |
| `"dinov2_B"` | dinov2_vitb14_reg | 768 | torch.hub |
| `"dinov2_L"` | dinov2_vitl14_reg | 1024 | torch.hub |
| `"dinov3_L"` | dinov3-vitl16-pretrain-lvd1689m | 1024 | HuggingFace transformers |
| `"dinov3_H"` | dinov3-vith16plus-pretrain-lvd1689m | 1280 | HuggingFace transformers |
| `"dinov3_7B"` | dinov3-vit7b16-pretrain-lvd1689m | 4096 | HuggingFace transformers |
| `"lseg"` | LSeg (demo_e200.ckpt) | 512 | Local checkpoint |
| `"maskclip"` | FeatUp MaskCLIP | 512 | torch.hub |

## Dataset Config Composition

Dataset configs compose a base + view sampler + optional dataset-specific overrides:

```yaml
# config/dataset/re10k.yaml
defaults:
  - base_dataset
  - view_sampler: bounded
  - optional view_sampler_dataset_specific_config@view_sampler: bounded_re10k

name: re10k
roots: [datasets/re10k]
input_image_shape: [224, 224]
original_image_shape: [360, 640]
```

The `@_group_` syntax in training configs allows multiple datasets:

```yaml
defaults:
  - /dataset@_group_.re10k: re10k
```

## CLI Override Examples

```bash
# Change learning rate
python -m src.main +training=gaussian_head optimizer.lr=5e-5

# Load a checkpoint
python -m src.main +training=gaussian_head checkpointing.load="path/to/ckpt"

# Switch to test mode
python -m src.main +evaluation=re10k mode=test

# Override view sampler for evaluation
python -m src.main +evaluation=re10k dataset/view_sampler@dataset.re10k.view_sampler=evaluation
```
