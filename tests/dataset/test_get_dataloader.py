import os

from dataset import get_dataloader


def test_dataloader(test_config):

    assert not os.path.isfile(test_config.DATASET.INPUT.STATS_FILE)

    train_dataloader = get_dataloader(test_config, "train")
    val_dataloader = get_dataloader(test_config, "val")
    test_dataloader = get_dataloader(test_config, "test")

    assert len(train_dataloader.dataset) == 2
    assert len(val_dataloader.dataset) == 2
    assert len(test_dataloader.dataset) == 1

    for batch in train_dataloader:
        pass

    # Test if stats json exists
    assert os.path.isfile(test_config.DATASET.INPUT.STATS_FILE)
    os.remove(test_config.DATASET.INPUT.STATS_FILE)
