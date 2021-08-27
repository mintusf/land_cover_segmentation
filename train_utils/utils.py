import torch
import random

import numpy as np
from torch.nn import Module, Softmax
from torch import Tensor
from torch.optim import Optimizer
from torchmetrics.functional import precision_recall, confusion_matrix

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
    checkpoint = torch.load(checkpoint_path)

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


def model_validation(model: Module, criterion: Module, val_dataloader: dict) -> dict:
    """Run a validation step on a whole val dataset and returns metrics
    Args:
        model (Module): Model to validate
        criterion (Module): Loss function
        val_dataloader (dict): Validation dataloader
    Returns:
        dict: Metrics:
              * precision
              * recall
              * f1
              * confusion_matrix
              * val_loss
    """
    with torch.no_grad():
        model.eval()
        val_loss = 0
        outputs = []
        targets = []
        for batch in val_dataloader:
            inputs, labels = batch["input"], batch["target"]

            # Forward propagation
            output = model(inputs)["out"]

            outputs.append(output)
            targets.append(labels)

            # Calc loss
            loss = criterion(output, labels)
            val_loss += loss.item()

        # Average loss
        val_loss /= len(val_dataloader)

    s = Softmax(dim=1)
    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)

    outputs = s(outputs)

    num_classes = outputs.shape[1]
    (precision_ave, recall_ave, f1_ave, confusion_matrix_whole,) = calc_metrics(
        outputs,
        targets,
        num_classes,
    )

    metrics = {
        "precision": precision_ave,
        "recall": recall_ave,
        "f1": f1_ave,
        "confusion_matrix": confusion_matrix_whole,
        "val_loss": val_loss,
    }

    return metrics


def calc_metrics(outputs: Tensor, targets: Tensor, num_classes: int) -> tuple:
    """Calculates segmentation metrics
    Args:
        outputs (Tensor): The model output, shape (N, C)
        targets (Tensor): The ground truth labels, shape (N)
        num_classes (int): Classes count
    Returns:
        tuple: contains metrics:
               * average precision
               * average recall
               * average f1 score
               * confusion matrix for all classes
    """

    precision_ave, recall_ave = precision_recall(
        preds=outputs,
        target=targets,
        average=None,
        num_classes=num_classes,
        mdmc_average="global",
    )

    precision_ave = np.nan_to_num(precision_ave.cpu()).mean()
    recall_ave = np.nan_to_num(recall_ave.cpu()).mean()

    f1_score_ave = np.divide(
        2 * precision_ave * recall_ave,
        (precision_ave + recall_ave),
    )

    f1_score_ave = np.nan_to_num(f1_score_ave)

    confusion_matrix_whole = confusion_matrix(outputs, targets, num_classes)

    return (
        np.float64(precision_ave),
        np.float64(recall_ave),
        np.float64(f1_score_ave),
        confusion_matrix_whole,
    )
