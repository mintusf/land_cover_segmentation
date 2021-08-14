import os

from torchvision.transforms import Compose
import numpy as np
import torch

from utils.utils import build_dataset_stats_json_from_cfg
from config.default import get_cfg_defaults
from dataset import PatchDataset
from dataset.transforms import get_transform


def test_dataset_init():
    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join("config", "tests.yml"))
    cfg.freeze()

    assert not os.path.isfile(cfg.DATASET.INPUT.STATS_FILE)

    build_dataset_stats_json_from_cfg(cfg)

    transform = get_transform(cfg)
    transforms = Compose([transform])

    dataset = PatchDataset(cfg, mode="train", transforms=transforms)

    # Calculate stats of transformed dataset
    means = torch.zeros((len(cfg.DATASET.INPUT.USED_CHANNELS)))
    stds = torch.zeros((len(cfg.DATASET.INPUT.USED_CHANNELS)))
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
        np.array(means), np.zeros((len(cfg.DATASET.INPUT.USED_CHANNELS))), decimal=4
    )
    np.testing.assert_almost_equal(
        np.array(stds), np.ones((len(cfg.DATASET.INPUT.USED_CHANNELS))), decimal=4
    )

    # Test if stats json exists
    assert os.path.isfile(cfg.DATASET.INPUT.STATS_FILE)
    os.remove(cfg.DATASET.INPUT.STATS_FILE)
