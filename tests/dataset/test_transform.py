import os
import sys
from torchvision.transforms import Compose
import numpy as np
import torch

from utils.utils import build_dataset_stats_json_from_cfg
from dataset import PatchDataset
from dataset.transforms import get_transform


def test_dataset_init(test_config):

    build_dataset_stats_json_from_cfg(test_config)

    transform = get_transform(test_config)
    transforms = Compose([transform])

    dataset = PatchDataset(test_config, mode="train", transforms=transforms)

    # Calculate stats of transformed dataset
    samples = []
    for sample in dataset:
        sample_input = sample["input"].cpu()
        samples.append(sample_input)

    merged = np.concatenate(samples, axis=1)
    global_means = np.mean(merged, axis=(1, 2))
    global_stds = np.std(merged, axis=(1, 2))

    # Compare with expected values
    np.testing.assert_almost_equal(
        np.array(global_means),
        np.zeros((len(test_config.DATASET.INPUT.USED_CHANNELS))),
        decimal=6,
    )
    np.testing.assert_almost_equal(
        np.array(global_stds),
        np.ones((len(test_config.DATASET.INPUT.USED_CHANNELS))),
        decimal=6,
    )
