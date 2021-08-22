import torch
from torch.nn import Module
from torch.nn.parallel import DataParallel

from config.default import CfgNode
from dataset.dataset_utils import get_channels_in_count, get_channels_out_count
from models.deeplab import create_deeplab


def get_model(cfg: CfgNode, device: str) -> Module:
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

    if "cuda" in device:
        if "all" in device:
            gpus_count = torch.cuda.device_count()
            assert gpus_count > 1, "No multi-gpu support"
            devices = [i for i in range(gpus_count)]
            model = DataParallel(model, device_ids=devices, output_device=devices[0])
            model.to("cuda:0")
        else:
            devices = device.split(":")[1].split(",")
            if len(devices) == 1:
                model.to(torch.device(device))
            else:
                devices = [int(d) for d in devices]
                model = DataParallel(
                    model, device_ids=devices, output_device=devices[0]
                )
                model.to(f"cuda:{devices[0]}")
    elif "cpu" in device:
        model.cpu()
    else:
        raise NotImplementedError

    return model
