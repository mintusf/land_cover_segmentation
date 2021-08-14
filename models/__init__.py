from torch.nn import Module

from config.default import CfgNode
from dataset.dataset_utils import get_channels_in_count, get_channels_out_count
from models.deeplab import create_deeplab


def get_model(cfg: CfgNode) -> Module:
    """Returns model Module

    Args:
        cfg (CfgNode): Config

    Raises:
        NotImplementedError: If model type is not supported

    Returns:
        Module: Model Module
    """
    if cfg.MODEL.TYPE == "DeepLab":
        channels_in = get_channels_in_count(cfg)
        channels_out = get_channels_out_count(cfg)
        return create_deeplab(channels_in, channels_out)
    else:
        raise NotImplementedError
