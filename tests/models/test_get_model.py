from torch.nn import Module

from models import get_model


def test_get_model(test_config):

    model = get_model(test_config, test_config.TRAIN.DEVICE)

    assert isinstance(model, Module)
