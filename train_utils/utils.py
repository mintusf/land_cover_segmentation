import torch
from typing import Tuple, Union
import random

import numpy as np
from torch import Tensor
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


def save_checkpoint(
    model: Module, epoch: int, optimizer, loss: float, cfg_path: str, save_path: str
) -> None:
    """Save checkpoint to file.
    Args:
        model (Module): Model to save
        epoch (int): Epoch number
        optimizer ([type]): Optimizer to save
        loss (float): Loss to save
        cfg_path (str): Path to config file
        save_path (str): Path to save checkpoint
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "cfg_path": cfg_path,
        },
        save_path,
    )


def load_checkpoint(checkpoint_path: str):
    """Load checkpoint from file.
    Args:
        checkpoint_path (str): Path to checkpoint file
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    epoch = checkpoint["epoch"]
    weights = checkpoint["model_state_dict"]
    optimizer = checkpoint["optimizer_state_dict"]
    loss = checkpoint["loss"]
    cfg_path = checkpoint["cfg_path"]

    return epoch, weights, optimizer, loss, cfg_path


def training_step(
    model: Module, optimizer: Optimizer, criterion: Module, batch: dict
) -> torch.Tensor:
    """Run a training step on a batch

    Args:
        model (Module): Model to train
        optimizer (Optimizer): Optimizer to use
        criterion (Module): Loss function
        batch (dict): Batch to train on

    Returns:
        torch.Tensor: Batch loss
    """
    model.train()
    inputs, labels = batch["input"], batch["target"]

    # Forward and backward propagations
    optimizer.zero_grad()
    outputs = model(inputs)["out"]
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    return loss


def model_validation(
    model: Module, criterion: Module, val_dataloader: dict, return_tensors: bool = False
) -> Union[float, Tuple[float, Tensor, Tensor]]:
    """Run a validation step on a whole val dataset and returns loss
    Args:
        model (Module): Model to validate
        criterion (Module): Loss function
        val_dataloader (dict): Validation dataloader
        return_preds (bool, optional): Whether to return predictions
    Returns:
        float: Validation loss
    """

    model.eval()
    val_loss = 0
    outputs_all = []
    inputs_all = []
    names_all = []
    for batch in val_dataloader:
        inputs, labels, names = batch["input"], batch["target"], batch["name"]

        # Forward propagation
        output = model(inputs)["out"]

        # Calc loss
        loss = criterion(output, labels)
        val_loss += loss.item()

        outputs_all.append(output)
        inputs_all.append(inputs)
        names_all.extend(names)

    # Average loss
    val_loss /= len(val_dataloader)

    if return_tensors:
        return val_loss, torch.cat(inputs_all, dim=0).detach(), torch.cat(outputs_all, dim=0).detach(), names_all
    else:
        return val_loss
