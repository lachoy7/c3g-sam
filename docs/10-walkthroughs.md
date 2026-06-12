# Code Flow Walkthroughs

Step-by-step traces through three representative scenarios. Each step cites `file::function`.

## Walkthrough 1: Gaussian Head Training Step

Config: `+training=gaussian_head`

### Setup (once)

1. **`src/main.py::train`** — Hydra resolves config. `load_typed_root_config` produces `RootCfg`.
2. **`src/main.py::train`** — `load_foundation_model(cfg)` returns `(None, None, None, None, 0)` because `cfg.train.reproj_model == 'none'`.
3. **`src/main.py::train`** — `cfg.model.encoder.feature_dim = 0` (no feature distillation).
4. **`src/model/encoder/__init__.py::get_encoder`** — Looks up `"vggt"` → instantiates `EncoderVGGT(cfg)`.
5. **`src/main.py::train`** — Loads VGGT pretrained weights (`pretrained_weights/model.pt`). Format 1: has `'model'` key → `checkpoint_filter_fn` → `encoder.load_state_dict`.
6. **`src/main.py::train`** — Constructs `ModelWrapper` with encoder, decoder, losses=[LossMse, LossLpips], vggt=None, dino=None.
7. **`src/dataset/data_module.py::DataModule`** — Creates DataModule with RE10K dataset config.

### Per-Step

1. **`DataModule::train_dataloader`** — Returns DataLoader over `DatasetRE10k` with `ViewSamplerBounded`.
2. **`DatasetRE10k::__iter__`** — Loads a `.torch` chunk, picks context/target via `ViewSamplerBounded.sample()` (2 context, 1 target), applies crop shim → yields `UnbatchedExample`.
3. **`ModelWrapper::training_step`** — Receives `batch: BatchedExample`.
4. **`ModelWrapper::training_step`** — `self.data_shim(batch)` applies encoder's normalize/patch shims.
5. **`ModelWrapper::training_step`** — `context_feature = None` (feature_dim == 0).
6. **`EncoderVGGT::forward`** — Runs VGGT backbone on context images → patch tokens (B×V×N×2048).
7. **`EncoderVGGT::forward`** — DPT head predicts 3D points from patch tokens.
8. **`EncoderVGGT::forward`** — Gaussian tokens (2048×2048) cross-attend to patch tokens via Transformer decoder (2 layers).
9. **`EncoderVGGT::forward`** — `UnifiedGaussianAdapter` converts raw outputs → `Gaussians(means, covariances, harmonics, opacities, feature=None)`.
10. **`ModelWrapper::training_step`** — Decoder forward: concatenates target + context extrinsics/intrinsics (context_view_loss=True).
11. **`DecoderSplattingCUDA::forward`** — Repeats Gaussians for all views, calls `render_cuda` → `DecoderOutput(color, depth, feature=None)`.
12. **`ModelWrapper::training_step`** — Constructs `target_gt` = concat(target_images, normalized_context_images).
13. **`ModelWrapper::training_step`** — Computes PSNR for logging.
14. **`LossMse::forward`** — `(prediction.color - target_gt)^2.mean()` × weight.
15. **`LossLpips::forward`** — If `global_step >= apply_after_step`: LPIPS(prediction.color, target_gt) × weight. Else: 0.
16. **`ModelWrapper::training_step`** — `feature_rendering_loss == 0` → skip feature loss.
17. **`ModelWrapper::training_step`** — Returns `total_loss`. Lightning handles backward + optimizer step.

## Walkthrough 2: RE10K Test Step (NVS Evaluation)

Config: `+evaluation=re10k mode=test dataset/view_sampler@dataset.re10k.view_sampler=evaluation`

### Setup

1. **`src/main.py::train`** — Same as above but `cfg.mode == "test"`.
2. **`src/main.py::train`** — Trainer created with `inference_mode=False` (align_pose=True).
3. **`src/main.py::train`** — `trainer.test(model_wrapper, datamodule, ckpt_path)` loads checkpoint.

### Per-Step

1. **`DatasetRE10k::__iter__`** — `ViewSamplerEvaluation.sample()` loads index from `evaluation_index_re10k.json` → returns fixed context/target indices.
2. **`ModelWrapper::test_step`** — `self.data_shim(batch)`.
3. **`ModelWrapper::test_step`** — Resizes context images to 224×224 if needed.
4. **`ModelWrapper::test_step`** — `context_feature = None` (feature_dim == 0 for Gaussian-only model).
5. **`EncoderVGGT::forward`** — Encodes context → `Gaussians`.
6. **`ModelWrapper::test_step`** — `test_cfg.align_pose == True` → calls `test_step_align(batch, gaussians)`.
7. **`ModelWrapper::test_step_align`** — Freezes encoder parameters.
8. **`ModelWrapper::test_step_align`** — Creates `cam_rot_delta` and `cam_trans_delta` as `nn.Parameter` (B×V×3).
9. **`ModelWrapper::test_step_align`** — Adam optimizer with `rot_opt_lr=0.005`, `trans_opt_lr=0.005`.
10. **`ModelWrapper::test_step_align`** — Loop for `pose_align_steps` iterations:
    - Render with current extrinsics + deltas
    - Compute MSE + LPIPS loss vs target images
    - Backward through deltas
    - `update_pose`: apply rotation (axis-angle → matrix) and translation deltas
    - Reset deltas to zero
    - Early stop if loss converges
11. **`ModelWrapper::test_step_align`** — Final render with optimized poses → returns `DecoderOutput`.
12. **`ModelWrapper::test_step`** — `compute_scores=True`:
    - `compute_psnr(rgb_gt, rgb_pred)`
    - `compute_ssim(rgb_gt, rgb_pred)`
    - `compute_lpips(rgb_gt, rgb_pred)`
13. **`ModelWrapper::test_step`** — Logs metrics, saves images to `outputs/test/<name>/<scene>/color/`.
14. **`ModelWrapper::test_step`** — If `save_compare=True`: creates comparison grid (context | GT | pred | error).
15. **`ModelWrapper::test_step`** — Renders Gaussian projections visualization.

## Walkthrough 3: Feature Head LSeg Training Step

Config: `+training=feature_head_lseg model.encoder.pretrained_weights="gaussian_ckpt"`

### Setup

1. **`src/main.py::train`** — `cfg.train.reproj_model == 'lseg'`.
2. **`src/model/load_foundation_model.py::load_foundation_model`** — Loads `LSegFeatureExtractor.from_pretrained('./pretrained_weights/demo_e200.ckpt', half_res=True)`. Returns `feature_dim=512`.
3. **`src/main.py::train`** — `cfg.model.encoder.feature_dim = 512`.
4. **`src/model/encoder/encoder_vggt.py::EncoderVGGT.__init__`** — Because `feature_dim > 0`, creates `InstillTransformer` (cross-attention decoder that also outputs features).
5. **`src/main.py::train`** — Loads Gaussian decoder checkpoint (Format 2: `state_dict` key, strip `encoder.` prefix). Missing keys include the new `InstillTransformer` parameters.

### Per-Step

1. **`ModelWrapper::training_step`** — `self.encoder.cfg.feature_dim == 512` → `context_feature = self.forward_foundation_model(batch['context']['image'])`.
2. **`ModelWrapper::forward_foundation_model`** — `reproj_model == 'lseg'`:

   ```python
   context_feature = self.lseg_feature_extractor.extract_features(images)
   # → (B, V, 512, H//2, W//2)
   ```

   Then interpolated to `(H//14, W//14)`.
3. **`EncoderVGGT::forward`** — Runs backbone → patch tokens.
4. **`EncoderVGGT::forward`** — `InstillTransformer` cross-attends Gaussian tokens to patch tokens AND `context_feature`. Outputs both Gaussian params and per-Gaussian features (B×2048×512).
5. **`EncoderVGGT::forward`** — Returns `Gaussians(..., feature=gaussian_features)`.
6. **`DecoderSplattingCUDA::forward`** — Renders color AND features. `feature_detach=True` means Gaussian positions/covariances are detached for feature rendering (geometry gradients only from color loss).
7. **`ModelWrapper::training_step`** — MSE loss on color.
8. **`ModelWrapper::training_step`** — LPIPS loss on color.
9. **`ModelWrapper::training_step`** — `feature_rendering_loss > 0`:
    - Extract LSeg features for ALL views (context + target): `forward_foundation_model(all_images, interpolate=False)` → `(B, V, 512, H//2, W//2)`.
    - Resize rendered Gaussian features to match: `F.interpolate(output.feature, size=(FH, FW))`.
    - L2-normalize both feature maps.
    - Compute `1 - cosine_similarity` → mean → multiply by weight (0.01).
10. **`ModelWrapper::training_step`** — `total_loss = mse + lpips + 0.01 * feature_loss`. Backward.

### At Test Time (ScanNet)

1. **`ModelWrapper::test_step`** — Renders Gaussian features for target views.
2. **`ModelWrapper::test_step`** — `lseg_feature_extractor.decode_feature(rendered_features, labelset)` → per-pixel class logits.
3. **`ModelWrapper::test_step`** — `argmax` → predicted segmentation map.
4. **`ModelWrapper::test_step`** — `self.miou(pred, target)` and `self.acc(pred, target)` using torchmetrics.
5. **`ModelWrapper::on_test_epoch_end`** — Reports mean IoU and mean accuracy.
