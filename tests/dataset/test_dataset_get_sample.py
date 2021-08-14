import torch

from dataset import PatchDataset


def test_dataset_get_sample(test_config):

    dataset = PatchDataset(test_config, mode="train")

    sample = dataset[0]

    # Test generated sample
    sample_input = sample["input"]
    sample_target = sample["target"]

    assert isinstance(sample_input, torch.Tensor)
    assert isinstance(sample_target, torch.Tensor)

    assert sample_input.dim() == 3
    assert sample_target.dim() == 2

    assert sample_input.shape[0] == len(test_config.DATASET.INPUT.USED_CHANNELS)
    assert sample_input.shape[1] == 256
    assert sample_input.shape[2] == 256
    assert sample_target.shape[0] == 256
    assert sample_target.shape[1] == 256
