import os

import torch
from torchvision.transforms import Compose

from models.deeplab import create_deeplab
from utils.io_utils import load_yaml
from dataset import PatchDataset
from dataset.transforms import get_transform


def test_deeplab_forward(test_config):
    channels_in = len(test_config.DATASET.INPUT.USED_CHANNELS)
    labels_config = load_yaml(test_config.DATASET.MASK.CONFIG)
    channels_out = len(labels_config["class2label"])

    assert channels_in == 4
    assert channels_out == 5

    model = create_deeplab(channels_in, channels_out)

    transform = get_transform(test_config)
    transforms = Compose([transform])
    dataset = PatchDataset(test_config, mode="train", transforms=transforms)

    sample_batch = torch.stack([dataset[0]["input"], dataset[1]["input"]], 0)

    pred = model(sample_batch)["out"]

    assert pred.dim() == 4
    assert pred.shape[0] == 2
    assert pred.shape[1] == channels_out
    assert pred.shape[2] == 256
    assert pred.shape[3] == 256

    # Test if stats json exists
    assert os.path.isfile(test_config.DATASET.INPUT.STATS_FILE)
    os.remove(test_config.DATASET.INPUT.STATS_FILE)
