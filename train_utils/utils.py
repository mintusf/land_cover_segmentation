import torch
import random

import numpy as np
from torch.nn import Module
from torch.optim import Optimizer

from config.default import CfgNode


def set_seeds(cfg: CfgNode) -> None:
    """Set random seeds

    Args:
        cfg (CfgNode): Config
    """
    seed = cfg.TRAIN.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def training_step(
    model: Module, optimizer: Optimizer, criterion: Module, batch: dict
) -> float:
    """Run a training step on a batch

    Args:
        model (Module): Model to train
        optimizer (Optimizer): Optimizer to use
        criterion (Module): Loss function
        batch (dict): Batch to train on

    Returns:
        float: Batch loss
    """
    model.train()
    inputs, labels = batch["input"], batch["target"]

    # zero the parameter gradients
    optimizer.zero_grad()

    # Forward propagation
    outputs = model(inputs)["out"]

    # Calc loss
    loss = criterion(outputs, labels)

    # Backward propagation
    loss.backward()

    # Optimize
    optimizer.step()

    return loss


def model_validation(model: Module, criterion: Module, val_dataloader: dict) -> float:
    """Run a validation step on a whole val dataset and returns loss
    Args:
        model (Module): Model to validate
        criterion (Module): Loss function
        val_dataloader (dict): Validation dataloader
    Returns:
        float: Validation loss
    """
    model.eval()
    val_loss = 0
    for batch in val_dataloader:
        inputs, labels = batch["input"], batch["target"]

        # Forward propagation
        outputs = model(inputs)["out"]

        # Calc loss
        loss = criterion(outputs, labels)
        val_loss += loss.item()

    # Average loss
    val_loss /= len(val_dataloader)
    return val_loss
