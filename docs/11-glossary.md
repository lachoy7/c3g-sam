# Glossary

## Core Concepts

**Gaussian Splatting (3DGS)**
A 3D scene representation using a set of 3D Gaussian primitives. Each Gaussian has a mean (position), covariance (shape/orientation), opacity, and color (spherical harmonics). Scenes are rendered by projecting Gaussians onto the image plane and alpha-compositing. See [3D Gaussian Splatting (Kerbl et al., 2023)](https://arxiv.org/abs/2308.14737).

**Spherical Harmonics (SH)**
A basis for representing view-dependent color on a sphere. SH degree 0 means a single RGB color per Gaussian (view-independent). Higher degrees capture specular effects. C3G uses degree 0 (`d_sh=1`).

**Novel View Synthesis (NVS)**
The task of rendering an image from a new camera viewpoint given one or more input images. C3G achieves this by predicting 3D Gaussians and rasterizing them from the target viewpoint.

**Feed-forward Reconstruction**
Predicting a 3D representation in a single forward pass of a neural network, as opposed to per-scene optimization (e.g., NeRF, standard 3DGS). C3G is feed-forward — it generalizes across scenes without test-time optimization of the 3D representation.

**Pose-free Reconstruction**
Reconstructing 3D scenes without requiring known camera poses as input. C3G predicts Gaussians directly from unposed images; camera poses are only needed for rendering (and can be optimized at test time).

## Models and Architectures

**Feature Distillation**
Training a model to reproduce the features of another (typically larger) model. In C3G, Gaussian features are trained to match VFM features via cosine similarity loss, enabling the compact Gaussian representation to carry semantic information.

**Vision Foundation Model (VFM)**
A large pretrained vision model that produces general-purpose features. C3G distills from LSeg, DINOv2, DINOv3, and VGGT.

**VGGT (Visual Geometry Grounded Transformer)**
A transformer model from Meta that jointly predicts camera poses, depth, and 3D points from multi-view images. C3G uses VGGT-1B as both a backbone encoder and a VFM for feature distillation. See [VGGT (Wang et al., 2025)](https://arxiv.org/abs/2503.11651).

**DUSt3R**
A model for dense unconstrained stereo 3D reconstruction from image pairs. Predicts pointmaps without requiring camera calibration. Inspired the pose-free approach in C3G. See [DUSt3R (Wang et al., 2024)](https://arxiv.org/abs/2312.14132).

**CroCo (Cross-view Completion)**
A self-supervised pretraining method for learning 3D-aware image representations by completing masked views. Used as the backbone in the NoPoSplat-based encoders. See [CroCo (Weinzaepfel et al., 2023)](https://arxiv.org/abs/2210.10716).

**NoPoSplat**
A method for pose-free generalizable 3D Gaussian splatting. C3G builds on its architecture for the encoder design. See [NoPoSplat (Ye et al., 2024)](https://arxiv.org/abs/2410.24207).

**pixelSplat**
A feed-forward method that predicts 3D Gaussians from posed image pairs. Provides the RE10K data preprocessing pipeline used by C3G. See [pixelSplat (Charatan et al., 2024)](https://arxiv.org/abs/2312.12337).

**MVSplat**
Multi-view extension of feed-forward Gaussian prediction. Also provides RE10K preprocessing. See [MVSplat (Chen et al., 2024)](https://arxiv.org/abs/2403.14627).

**DPT Head (Dense Prediction Transformer)**
A decoder architecture that fuses multi-scale transformer features for dense prediction tasks (depth, segmentation). Used in C3G for depth/point prediction from patch tokens. See [DPT (Ranftl et al., 2021)](https://arxiv.org/abs/2103.13413).

**Gaussian Adapter**
A module (`src/model/encoder/common/gaussian_adapter.py`) that converts raw network outputs into valid Gaussian parameters — applying sigmoid to opacities, constructing covariance matrices from scale/rotation predictions, etc.

## Data and Sampling

**View Sampler**
A strategy for selecting which frames in a video sequence become context views (input) and target views (supervision). Options: bounded, evaluation, all, arbitrary.

**Context View**
An input image provided to the encoder. The model observes these views and predicts 3D Gaussians from them. Typically 2 views (or up to 24 in multi-view mode).

**Target View**
A view used for supervision during training or evaluation during testing. The model renders Gaussians from the target viewpoint and compares against the ground-truth image.

**Epipolar Line**
The projection of a ray from one camera onto another camera's image plane. Used in `src/geometry/epipolar_lines.py` for computing view overlap and in the evaluation index generator.

## Metrics

**PSNR (Peak Signal-to-Noise Ratio)**
Measures pixel-level reconstruction quality. Higher is better. Computed as `-10 * log10(MSE)` where MSE is mean squared error between predicted and ground-truth images (both in [0,1]).

**SSIM (Structural Similarity Index)**
Measures structural similarity between images considering luminance, contrast, and structure. Range [0, 1], higher is better. Uses 11×11 Gaussian-weighted windows.

**LPIPS (Learned Perceptual Image Patch Similarity)**
Measures perceptual similarity using deep features (VGG network). Lower is better. More aligned with human perception than PSNR/SSIM.

**mIoU (mean Intersection over Union)**
Standard metric for semantic segmentation. Computes IoU per class and averages. Used for ScanNet/Replica scene understanding evaluation. Ignores class 0 (background/unknown).

**Accuracy**
Fraction of correctly classified pixels. Also ignores class 0.

**Pose AUC**
Area Under the Curve for pose error at thresholds [5°, 10°, 20°]. Used in pose estimation evaluation. Higher is better.
