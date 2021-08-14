from torch import Tensor

from train_utils import training_step, model_validation


def test_train_step(module_dict):

    model = module_dict["model"]
    optimizer = module_dict["optimizer"]
    criterion = module_dict["criterion"]

    train_dataloader = module_dict["train_dataloader"]

    for batch in train_dataloader:
        loss = training_step(model, optimizer, criterion, batch)
        assert isinstance(loss, Tensor)


def test_val_step(module_dict):

    model = module_dict["model"]
    criterion = module_dict["criterion"]

    val_dataloader = module_dict["val_dataloader"]

    loss = model_validation(model, criterion, val_dataloader)

    assert isinstance(loss, float)
