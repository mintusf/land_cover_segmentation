import os

import torch

from config.default import get_cfg_defaults
from dataset import PatchDataset


def test_dataset_get_sample():
    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join("config", "tests.yml"))
    cfg.freeze()

    dataset = PatchDataset(cfg)

    sample = dataset[0]

    # Test generated sample
    sample_input = sample["input"]
    sample_target = sample["target"]

    assert isinstance(sample_input, torch.Tensor)
    assert isinstance(sample_target, torch.Tensor)

    assert sample_input.dim() == 3
    assert sample_target.dim() == 3

    assert sample_input.shape[0] == len(cfg.DATASET.INPUT.USED_CHANNELS)
    assert sample_input.shape[1] == 256
    assert sample_input.shape[2] == 256
    assert sample_target.shape[0] == 1
    assert sample_target.shape[1] == 256
    assert sample_target.shape[2] == 256
