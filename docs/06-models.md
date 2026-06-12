# Models and Foundation Models

## Encoder Variants

### EncoderVGGT (`src/model/encoder/encoder_vggt.py`)

The primary encoder used in C3G. Takes multi-view images and predicts 2048 Gaussians.

**Config** (`EncoderVGGTCfg`):

```python
@dataclass
class EncoderVGGTCfg:
    name: Literal["vggt"]
    num_gaussians: int          # 2048
    backbone: BackboneCfg       # vggt_multi
    gaussian_adapter: GaussianAdapterCfg
    gs_params_head_type: str    # "dpt_gs"
    pose_free: bool             # True
    freeze_backbone: bool       # False
    decoder_depth: int          # 2 (Transformer decoder layers)
    gaussians_per_token: int    # 1
    gaussian_feature_dim: int   # 0 or 512/768/1024/2048
    feature_dim: int            # set automatically from VFM
```

**Architecture**:

1. **VGGT Backbone** (`backbone_vggt.py`): Multi-view ViT with aggregation. Produces 2048-dim patch tokens.
2. **DPT Head**: Predicts dense depth/3D points from patch tokens (frozen after pretraining).
3. **Gaussian Tokens**: Learnable parameters `(2048, 2048)` initialized randomly.
4. **Transformer Decoder** (`Transformer` or `InstillTransformer`): Cross-attention between Gaussian tokens and patch tokens.
5. **GaussianAdapter**: Converts raw outputs to proper Gaussian parameters.

When `feature_dim > 0`, an `InstillTransformer` is used instead of a plain `Transformer`, adding a feature prediction head.

### EncoderNoPoSplat (`src/model/encoder/encoder_noposplat.py`)

The 2-view encoder based on the [NoPoSplat](https://github.com/cvg/NoPoSplat) architecture.

**Architecture**:

1. **CroCo Backbone** (`backbone_croco.py`): Cross-view completion network for 2-view input.
2. **DPT Head** (head1/head2): Per-view depth prediction.
3. **GS Params Head**: Predicts Gaussian parameters per pixel or per token.
4. **GaussianAdapter**: Same as VGGT variant.

### EncoderNoPoSplatMulti (`src/model/encoder/encoder_noposplat_multi.py`)

Multi-view extension of EncoderNoPoSplat using `backbone_croco_multiview.py`.

## Decoder

### DecoderSplattingCUDA (`src/model/decoder/decoder_splatting_cuda.py`)

The only decoder in the system. Performs differentiable Gaussian rasterization.

**Config** (`DecoderSplattingCUDACfg`):

```python
@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]
    background_color: list[float]     # [0, 0, 0]
    make_scale_invariant: bool        # True
    low_pass_filter: float            # 0.3 or 10.0 (decreases during training)
    decrease_lpf_step: int            # step interval to decrease LPF
    feature_detach: bool              # detach geometry from feature gradients
```

**Forward pass**:

1. Repeats Gaussians for each target view
2. Calls `render_cuda` from `cuda_splatting.py` (wraps gsplat)
3. Returns `DecoderOutput(color, depth, feature)`

**Scale invariance**: When enabled, normalizes Gaussian means before rendering so the representation is independent of absolute scale.

**Low-pass filter**: Anti-aliasing parameter. Starts high (10.0) and decreases by 3× every `decrease_lpf_step` steps until reaching 0.3.

**Pose optimization support**: Accepts `cam_rot_delta` and `cam_trans_delta` tensors for differentiable test-time pose refinement.

## Gaussians Dataclass

```python
@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]           # (B, 2048, 3)
    covariances: Float[Tensor, "batch gaussian dim dim"] # (B, 2048, 3, 3)
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]    # (B, 2048, 3, 1) for SH degree 0
    opacities: Float[Tensor, "batch gaussian"]           # (B, 2048)
    feature: Float[Tensor, "batch gaussian feature_dim"] # (B, 2048, D) or None
```

## GaussianAdapter (`src/model/encoder/common/gaussian_adapter.py`)

Converts raw network outputs into valid Gaussian parameters:

- **Means**: Direct 3D coordinates (or from depth + unprojection)
- **Covariances**: Constructed from predicted scales and rotations (quaternion → rotation matrix → covariance)
- **Harmonics**: Raw SH coefficients (degree 0 = single color per Gaussian)
- **Opacities**: Sigmoid activation with optional clamping
- **Features**: Direct output from feature head

`UnifiedGaussianAdapter` is used in pose-free mode — it handles the case where 3D positions are predicted directly without requiring known camera poses.

## DPT Heads (`src/model/encoder/heads.py`)

Dense Prediction Transformer heads for converting patch tokens to dense predictions:

- **DPTHead**: Standard DPT for depth/point prediction (output_dim=3)
- **GSDPTHead**: Variant for Gaussian parameter prediction (output_dim=raw_gs_dim)

Both use multi-scale feature fusion with learned upsampling.

## Transformer / InstillTransformer (`src/model/encoder/common/gmae.py`)

Custom Transformer decoder blocks:

- **Transformer**: Standard cross-attention decoder. Gaussian tokens attend to image patch tokens.
- **InstillTransformer**: Extended version that also predicts per-Gaussian features for VFM distillation. Adds a feature output head alongside the Gaussian parameter head.

Both use:

- Multi-head cross-attention (16 heads, dim_head=128 for 2048-dim)
- Feed-forward MLP (2× expansion)
- Pre-norm (LayerNorm before attention)

## Foundation Model Loading (`src/model/load_foundation_model.py`)

Dispatch function that loads the appropriate VFM based on `cfg.train.reproj_model`:

```python
def load_foundation_model(cfg):
    # Returns: (vggt, dino, lseg_feature_extractor, clip_model, feature_dim)
```

| Model | Loading Method | Frozen | Notes |
|-------|---------------|--------|-------|
| VGGT | `torch.hub.load_state_dict_from_url` | Yes | Full VGGT-1B model |
| DINOv2 | `torch.hub.load('facebookresearch/dinov2', ...)` | Yes | ViT-B or ViT-L with registers |
| DINOv3 | `AutoModel.from_pretrained(...)` | Yes | HuggingFace transformers |
| LSeg | `LSegFeatureExtractor.from_pretrained(...)` | Yes | Local checkpoint |
| MaskCLIP | `torch.hub.load("mhamilton723/FeatUp", ...)` | Yes | FeatUp upsampled CLIP |

All foundation models are frozen (no gradients) and used only for feature extraction during training.

## External References

- [VGGT: Visual Geometry Grounded Transformer](https://github.com/facebookresearch/vggt) — Backbone and VFM
- [NoPoSplat](https://github.com/cvg/NoPoSplat) — Pose-free Gaussian prediction architecture
- [CroCo](https://github.com/naver/croco) — Cross-view completion pretraining
- [DINOv2](https://github.com/facebookresearch/dinov2) — Self-supervised ViT features
- [LSeg](https://github.com/isl-org/lang-seg) — Language-driven segmentation
- [gsplat](https://github.com/nerfstudio-project/gsplat) — Differentiable Gaussian splatting
- [FeatUp](https://github.com/mhamilton723/FeatUp) — Feature upsampling for CLIP/DINO
