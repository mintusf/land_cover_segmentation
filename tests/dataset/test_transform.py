import os

from torchvision.transforms import Compose
import numpy as np
import torch

from utils.utils import build_dataset_stats_json
from config.default import get_cfg_defaults
from dataset import PatchDataset
from dataset.transforms import get_transform
from utils.io_utils import get_lines_from_txt


def test_dataset_init():
    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join("config", "tests.yml"))
    cfg.freeze()

    assert not os.path.isfile(cfg.DATASET.INPUT.STATS_FILE)

    dataset_list = get_lines_from_txt(cfg.DATASET.LIST)
    build_dataset_stats_json(
        dataset_list,
        cfg.DATASET.ROOT,
        cfg.DATASET.INPUT.SENSOR,
        cfg.DATASET.INPUT.CHANNELS,
        cfg.DATASET.INPUT.STATS_FILE,
    )

    transform = get_transform(cfg)
    transforms = Compose([transform])

    dataset = PatchDataset(cfg, transforms=transforms)

    # Calculate stats of transformed dataset
    means = torch.Tensor([0, 0, 0])
    stds = torch.Tensor([0, 0, 0])
    for sample in dataset:
        sample_input = sample["input"]
        image_means = torch.mean(sample_input, dim=[1, 2])
        image_stds = torch.std(sample_input, dim=[1, 2])
        means += image_means
        stds += image_stds

    means = means / len(dataset)
    stds = stds / len(dataset)

    # Compare with expected values
    np.testing.assert_almost_equal(np.array(means), np.array([0, 0, 0]), decimal=4)
    np.testing.assert_almost_equal(np.array(stds), np.array([1, 1, 1]), decimal=4)

    # Test if stats json exists
    assert os.path.isfile(cfg.DATASET.INPUT.STATS_FILE)
    os.remove(cfg.DATASET.INPUT.STATS_FILE)
