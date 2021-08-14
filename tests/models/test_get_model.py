import os

from torch.nn import Module

from models import get_model
from config.default import get_cfg_from_file


def test_get_model():
    cfg = get_cfg_from_file(os.path.join("config", "tests.yml"))

    model = get_model(cfg)

    assert isinstance(model, Module)
