from dataset import get_dataloader
from tests.conftest import with_class_json


@with_class_json
def test_dataloader(test_config):

    train_dataloader = get_dataloader(test_config, "train")
    val_dataloader = get_dataloader(test_config, "val")
    test_dataloader = get_dataloader(test_config, "test")

    assert len(train_dataloader.dataset) == 2
    assert len(val_dataloader.dataset) == 2
    assert len(test_dataloader.dataset) == 1

    for batch in train_dataloader:
        pass
