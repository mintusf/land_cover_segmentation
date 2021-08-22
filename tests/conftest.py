from dataset.transforms import get_transform
import os
import pytest

from config.default import get_cfg_from_file
from train_utils import get_loss, get_optimizer, get_lr_scheduler
from models import get_model
from dataset import get_dataloader

test_config_path = os.path.join("config", "tests.yml")


@pytest.fixture(scope="session")
def test_config():
    cfg = get_cfg_from_file(test_config_path)
    return cfg


@pytest.fixture(scope="session")
def module_dict():

    cfg = get_cfg_from_file(test_config_path)

    model = get_model(cfg, cfg.TRAIN.DEVICE)
    optimizer = get_optimizer(model, cfg)
    criterion = get_loss(cfg)
    lr_scheduler = get_lr_scheduler(optimizer, cfg)

    transforms = get_transform(cfg)
    train_dataloader = get_dataloader(cfg, "train")
    val_dataloader = get_dataloader(cfg, "val")

    out_dict = {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "lr_scheduler": lr_scheduler,
        "transforms": transforms,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
    }

    return out_dict


def pytest_sessionfinish():
    cfg = get_cfg_from_file(test_config_path)
    assert os.path.isfile(cfg.DATASET.INPUT.STATS_FILE)
    os.remove(cfg.DATASET.INPUT.STATS_FILE)
    os.remove("tests/utils/test_vis.png")
