import torch
from torchvision.transforms import CenterCrop, Compose

from src.model.sam.loader import load_sam_encoder

DINOV3_MODELS = {
    "dinov3_L": ("facebook/dinov3-vitl16-pretrain-lvd1689m", 1024),
    "dinov3_H": ("facebook/dinov3-vith16plus-pretrain-lvd1689m", 1280),
    "dinov3_7B": ("facebook/dinov3-vit7b16-pretrain-lvd1689m", 4096),
}


def load_foundation_model(cfg, *, skip_sam_encoder: bool = False):
    vggt, dino, lseg_feature_extractor, clip_model, sam_encoder = (
        None,
        None,
        None,
        None,
        None,
    )
    feature_dim = 0
    if "vggt" in cfg.train.reproj_model:
        from src.model.encoder.backbone.vggt.vggt import VGGT

        vggt = VGGT()
        msg = vggt.load_state_dict(
            torch.hub.load_state_dict_from_url(
                "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
            )
        )
        vggt.eval()
        for param in vggt.parameters():
            param.requires_grad = False

        feature_dim = 2048

    elif "dinov3" in cfg.train.reproj_model:
        from transformers import AutoImageProcessor, AutoModel

        pretrained_model_name, feature_dim = DINOV3_MODELS[cfg.train.reproj_model]

        processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        dino_model = AutoModel.from_pretrained(
            pretrained_model_name,
        )

        for param in dino_model.parameters():
            param.requires_grad = False

        dino = {"model": dino_model, "processor": processor}

    elif "dinov2" in cfg.train.reproj_model:
        if "dinov2_B" == cfg.train.reproj_model:
            dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
            feature_dim = 768
        elif "dinov2_L" == cfg.train.reproj_model:
            dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
            feature_dim = 1024

        for param in dino_model.parameters():
            param.requires_grad = False

        dino = {"model": dino_model, "processor": None}

    elif "lseg" in cfg.train.reproj_model:
        from src.model.lseg import LSegFeatureExtractor

        lseg_feature_extractor = LSegFeatureExtractor.from_pretrained(
            "./pretrained_weights/demo_e200.ckpt", half_res=True
        )
        feature_dim = 512

        for param in lseg_feature_extractor.parameters():
            param.requires_grad = False

    elif "sam" in cfg.train.reproj_model:
        feature_dim = 256
        if not skip_sam_encoder:
            model_variant = getattr(cfg.train, "sam_model_variant", "sam_vit_h")
            checkpoint_path = getattr(
                cfg.train, "sam_checkpoint", "./pretrained_weights/sam_vit_h.pth"
            )
            sam_encoder, feature_dim = load_sam_encoder(model_variant, checkpoint_path)

    if cfg.train.reproj_model == "maskclip":
        upsampler = torch.hub.load("mhamilton723/FeatUp", "maskclip", use_norm=False)

        for param in upsampler.parameters():
            param.requires_grad = False

        clip_model = {"model": upsampler}
        feature_dim = 512

    return vggt, dino, lseg_feature_extractor, clip_model, sam_encoder, feature_dim
