import torch
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel

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
        model = create_deeplab(channels_in, channels_out)
    else:
        raise NotImplementedError

    if "cuda" in cfg.TRAIN.DEVICE and "all" not in cfg.TRAIN.DEVICE:
        model.to(torch.device(cfg.TRAIN.DEVICE))
    elif "cpu" in cfg.TRAIN.DEVICE:
        model.cpu()
    elif cfg.TRAIN.DEVICE == "cuda:all":
        assert torch.cuda.device_count() > 1, "No multi-gpu support"
        torch.distributed.init_process_group()
        model = DistributedDataParallel(model)
    else:
        raise NotImplementedError

    return model
