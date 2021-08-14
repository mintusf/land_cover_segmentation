import os

from config.default import get_cfg_defaults
from dataset import PatchDataset


def test_dataset_init():
    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join("config", "tests.yml"))
    cfg.freeze()

    dataset = PatchDataset(cfg, mode="train")

    assert len(dataset) == 2
