import os

from torchvision.transforms import Compose
import numpy as np
import torch

from utils.utils import build_dataset_stats_json_from_cfg
from dataset import PatchDataset
from dataset.transforms import get_transform
from tests.conftest import with_class_json

@with_class_json
def test_dataset_init(test_config):

    build_dataset_stats_json_from_cfg(test_config)

    transform = get_transform(test_config)
    transforms = Compose([transform])

    dataset = PatchDataset(test_config, mode="train", transforms=transforms)

    # Calculate stats of transformed dataset
    means = torch.zeros((len(test_config.DATASET.INPUT.USED_CHANNELS)))
    stds = torch.zeros((len(test_config.DATASET.INPUT.USED_CHANNELS)))
    for sample in dataset:
        sample_input = sample["input"]
        image_means = torch.mean(sample_input, dim=[1, 2])
        image_stds = torch.std(sample_input, dim=[1, 2])
        means += image_means
        stds += image_stds

    means = means / len(dataset)
    stds = stds / len(dataset)

    # Compare with expected values
    np.testing.assert_almost_equal(
        np.array(means),
        np.zeros((len(test_config.DATASET.INPUT.USED_CHANNELS))),
        decimal=4,
    )
    np.testing.assert_almost_equal(
        np.array(stds),
        np.ones((len(test_config.DATASET.INPUT.USED_CHANNELS))),
        decimal=4,
    )
