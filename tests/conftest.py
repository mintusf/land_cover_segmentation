from dataset.transforms import get_transform
import os
import pytest
from shutil import rmtree

from config.default import get_cfg_from_file
from train_utils import get_loss, get_optimizer, get_lr_scheduler, save_checkpoint
from models import get_model
from dataset import get_dataloader

test_config_path = os.path.join("config", "tests.yml")
checkpoint_save_path = os.path.join("tests", "train_utils", "test_checkpoint")
infer_directory = os.path.join("tests", "test_masks")


@pytest.fixture(scope="session")
def test_config():
    cfg = get_cfg_from_file(test_config_path)
    return cfg


@pytest.fixture(scope="session")
def test_checkpoint(test_config):

    model = get_model(test_config, test_config.TRAIN.DEVICE)
    optimizer = get_optimizer(model, test_config)
    epoch = 1
    loss = 1.0

    save_checkpoint(model, epoch, optimizer, loss, test_config, checkpoint_save_path)

    return {
        "path": checkpoint_save_path,
        "epoch": epoch,
        "loss": loss,
        "weights": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg_path": test_config,
    }


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
        "cfg_path": test_config_path,
        "infer_directory": infer_directory,
    }

    return out_dict


def pytest_sessionfinish():
    cfg = get_cfg_from_file(test_config_path)
    assert os.path.isfile(cfg.DATASET.INPUT.STATS_FILE)
    os.remove(cfg.DATASET.INPUT.STATS_FILE)
    os.remove("tests/utils/test_vis.png")
    os.remove(checkpoint_save_path)
    rmtree(infer_directory)
