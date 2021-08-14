from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import get_model
from train_utils import get_loss, get_optimizer, get_lr_scheduler


def test_get_loss(test_config):

    loss = get_loss(test_config)

    assert isinstance(loss, CrossEntropyLoss)


def test_get_optimizer(test_config):

    model = get_model(test_config)
    optimizer = get_optimizer(model, test_config)

    assert isinstance(optimizer, Optimizer)


def test_get_lr_scheduler(test_config):

    model = get_model(test_config)
    optimizer = get_optimizer(model, test_config)

    scheduler = get_lr_scheduler(optimizer, test_config)

    assert isinstance(scheduler, ReduceLROnPlateau)
