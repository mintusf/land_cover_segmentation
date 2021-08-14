import os

import torch

from models import get_model
from train_utils import get_optimizer, save_checkpoint, load_checkpoint


def test_checkpoint(test_config):

    model = get_model(test_config)
    optimizer = get_optimizer(model, test_config)
    epoch = 1
    loss = 1.0
    save_path = os.path.join("tests", "train_utils", "test_checkpoint")

    save_checkpoint(model, epoch, optimizer, loss, test_config, save_path)

    (
        loaded_epoch,
        loaded_weights,
        loaded_optimizer,
        loaded_loss,
        loaded_cfg,
    ) = load_checkpoint(save_path)

    assert loaded_epoch == epoch

    compared_layer = "backbone.conv1.weight"
    assert torch.all(
        torch.eq(
            loaded_weights[compared_layer],
            model.state_dict()[compared_layer],
        )
    )
    assert loaded_optimizer == optimizer.state_dict()
    assert loaded_loss == loss
    assert loaded_cfg == test_config

    os.remove(save_path)
