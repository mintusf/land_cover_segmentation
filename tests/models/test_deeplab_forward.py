import os

import torch
from torchvision.transforms import Compose

from utils.utils import build_dataset_stats_json_from_cfg
from models.deeplab import create_deeplab
from config.default import get_cfg_defaults
from utils.io_utils import load_yaml
from dataset import PatchDataset
from dataset.transforms import get_transform


def test_deeplab_forward():
    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join("config", "tests.yml"))
    cfg.freeze()
    channels_in = len(cfg.DATASET.INPUT.USED_CHANNELS)
    labels_config = load_yaml(cfg.DATASET.MASK.CONFIG)
    channels_out = len(labels_config["class2label"])

    assert channels_in == 4
    assert channels_out == 5

    model = create_deeplab(channels_in, channels_out)

    build_dataset_stats_json_from_cfg(cfg)
    transform = get_transform(cfg)
    transforms = Compose([transform])
    dataset = PatchDataset(cfg, transforms)

    sample_batch = torch.stack([dataset[0]["input"], dataset[1]["input"]], 0)

    pred = model(sample_batch)["out"]

    assert pred.dim() == 4
    assert pred.shape[0] == 2
    assert pred.shape[1] == channels_out
    assert pred.shape[2] == 256
    assert pred.shape[3] == 256

    # Test if stats json exists
    assert os.path.isfile(cfg.DATASET.INPUT.STATS_FILE)
    os.remove(cfg.DATASET.INPUT.STATS_FILE)
