from .loss import Loss
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_segmentation import LossSegmentation, LossSegmentationCfgWrapper

LOSSES = {
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossSegmentationCfgWrapper: LossSegmentation,
}

LossCfgWrapper = LossLpipsCfgWrapper | LossMseCfgWrapper | LossSegmentationCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
