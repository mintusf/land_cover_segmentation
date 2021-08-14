import os

from config.default import get_cfg_from_file
from dataset import get_dataloader


def test_dataloader():

    cfg = get_cfg_from_file(os.path.join("config", "tests.yml"))

    assert not os.path.isfile(cfg.DATASET.INPUT.STATS_FILE)

    train_dataloader = get_dataloader(cfg, "train")
    val_dataloader = get_dataloader(cfg, "val")
    test_dataloader = get_dataloader(cfg, "test")

    assert len(train_dataloader.dataset) == 2
    assert len(val_dataloader.dataset) == 2
    assert len(test_dataloader.dataset) == 1

    for batch in train_dataloader:
        pass

    # Test if stats json exists
    assert os.path.isfile(cfg.DATASET.INPUT.STATS_FILE)
    os.remove(cfg.DATASET.INPUT.STATS_FILE)
