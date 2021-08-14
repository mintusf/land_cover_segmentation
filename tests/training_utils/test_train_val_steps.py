import os
from train_utils import training_step, model_validation, get_loss, get_optimizer
from models import get_model
from dataset import get_dataloader
from tests.conftest import with_class_json

@with_class_json
def test_train_step(test_config):

    model = get_model(test_config)
    optimizer = get_optimizer(model, test_config)
    criterion = get_loss(test_config)

    train_dataloader = get_dataloader(test_config, "train")

    for batch in train_dataloader:
        training_step(model, optimizer, criterion, batch)

@with_class_json
def test_val_step(test_config):

    model = get_model(test_config)
    criterion = get_loss(test_config)

    val_dataloader = get_dataloader(test_config, "val")

    loss = model_validation(model, criterion, val_dataloader)

    assert isinstance(loss, float)
