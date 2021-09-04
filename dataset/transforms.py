import os
from typing import Dict, Tuple

import torch
from yacs.config import CfgNode

from utils.io_utils import load_json
from utils.utilities import build_dataset_stats_json_from_cfg


class NormalizeSample(object):
    def __init__(
        self,
        dataset_mean: Tuple[float],
        dataset_std: Tuple[float],
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ):
        """Standardize sample's input channel-wise.
        Args:
            dataset_mean (float, optional): Dataset mean.
            dataset_std (float, optional): Dataset std.
            target_mean (float, optional): Target mean. Defaults to 0.0.
            target_std (float, optional): Target std. Defaults to 1.0.
        """
        self.target_mean = target_mean
        self.target_std = target_std
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Standardize sample's input channel-wise.
        Args:
            sample (Dict[torch.Tensor]): sample to be normalized
        Returns:
            Dict[torch.Tensor]: sample with normalized raster
        """
        sample_input = sample["input"]

        for channel in range(sample_input.shape[0]):
            sample_input[channel, :, :] = (
                sample_input[channel, :, :] - self.dataset_mean[channel]
            ) / self.dataset_std[channel]

        sample["input"] = sample_input

        return sample


def get_transform(cfg: CfgNode) -> NormalizeSample:
    """Gets transform function. Builds stats dict if not existing.

    Args:
        cfg (CfgNode): Config

    Returns:
        NormalizeSample: NormalizeSample object
    """
    channels = cfg.DATASET.INPUT.CHANNELS
    used_channels = cfg.DATASET.INPUT.USED_CHANNELS
    stats_file = cfg.DATASET.INPUT.STATS_FILE

    if not os.path.isfile(stats_file):
        build_dataset_stats_json_from_cfg(cfg)
    stats = load_json(stats_file)
    means = [stats["means"][channels[channel]] for channel in used_channels]
    stds = [stats["stds"][channels[channel]] for channel in used_channels]

    transform = NormalizeSample(dataset_mean=means, dataset_std=stds)

    return transform
