import os
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config.default import get_cfg_from_file
from models import get_model
from train_utils import get_loss, get_optimizer, get_lr_scheduler


def test_get_loss():
    cfg = get_cfg_from_file(os.path.join("config", "tests.yml"))

    loss = get_loss(cfg)

    assert isinstance(loss, CrossEntropyLoss)


def test_get_optimizer():
    cfg = get_cfg_from_file(os.path.join("config", "tests.yml"))

    model = get_model(cfg)
    optimizer = get_optimizer(model, cfg)

    assert isinstance(optimizer, Optimizer)


def test_get_lr_scheduler():
    cfg = get_cfg_from_file(os.path.join("config", "tests.yml"))

    model = get_model(cfg)
    optimizer = get_optimizer(model, cfg)

    scheduler = get_lr_scheduler(optimizer, cfg)

    assert isinstance(scheduler, ReduceLROnPlateau)