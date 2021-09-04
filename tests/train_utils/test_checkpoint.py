import os

import torch

from models import get_model
from train_utils import get_optimizer, save_checkpoint, load_checkpoint


def test_checkpoint(test_config, test_checkpoint):

    (
        loaded_epoch,
        loaded_weights,
        loaded_optimizer,
        loaded_loss,
        loaded_cfg,
    ) = load_checkpoint(test_checkpoint["path"], test_config.TRAIN.DEVICE)

    assert loaded_epoch == test_checkpoint["epoch"]

    compared_layer = "backbone.conv1.weight"
    assert torch.all(
        torch.eq(
            loaded_weights[compared_layer],
            test_checkpoint["weights"][compared_layer],
        )
    )
    assert loaded_optimizer == test_checkpoint["optimizer_state_dict"]
    assert loaded_loss == test_checkpoint["loss"]
    assert loaded_cfg == test_checkpoint["cfg_path"]
