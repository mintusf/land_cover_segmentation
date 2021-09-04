import logging
import os
from typing import Dict, Tuple, Union

import torch
import random

import numpy as np
from torch import Tensor
from torch.nn import Module, Softmax
from torch.optim import Optimizer
from torchmetrics.functional import confusion_matrix

from config.default import CfgNode, get_cfg_from_file


logger = logging.getLogger("global")


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


def load_checkpoint(checkpoint_path: str, device: str):
    """Load checkpoint from file.
    Args:
        checkpoint_path (str): Path to checkpoint file
        device (str): Device to load checkpoint on
    """
    if "cpu" in device:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    else:
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


def model_validation(
    model: Module,
    criterion: Module,
    val_dataloader: dict,
    return_ave: bool = True,
) -> dict:
    """Run a validation step on a whole val dataset and returns metrics
    Args:
        model (Module): Model to validate
        criterion (Module): Loss function
        val_dataloader (dict): Validation dataloader
        return_tensors (bool, optional): Whether to return tensors
        return_ave (bool, optional): Whether to return average metrics.
                                     Metrics for each class returned if False.
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
        s = Softmax(dim=1)
        num_classes = len(val_dataloader.dataset.mask_config["class2label"])
        confusion_matrix_whole = np.zeros((num_classes, num_classes))
        for batch in val_dataloader:
            inputs, labels = batch["input"], batch["target"]

            # Forward propagation
            outputs = model(inputs)["out"]

            # Calc loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Calc metrics
            outputs = s(outputs)
            confusion_matrix_batch = confusion_matrix(outputs, labels, num_classes)
            confusion_matrix_whole += confusion_matrix_batch.cpu().numpy()

        # Average loss
        val_loss /= len(val_dataloader)

        # Calcualte recall precision and f1
        recall_ave, precision_ave, f1_ave = calc_metrics(
            confusion_matrix_whole, return_ave
        )

    metrics = {
        "precision": precision_ave,
        "recall": recall_ave,
        "f1": f1_ave,
        "confusion_matrix": confusion_matrix_whole,
        "val_loss": val_loss,
    }

    return metrics


def calc_metrics(confusion_matrix: Tensor, return_ave: bool = True) -> Tuple:
    """Calculates segmentation metrics
    Args:
        confusion_matrix (Tensor): Confusion matrix
        return_ave (bool, optional): Whether to return average metrics
    Returns:
        tuple: contains metrics:
               * average precision
               * average recall
               * average f1 score
    """

    # Calculate metrics
    recall = np.array(
        [
            confusion_matrix[i, i] / np.sum(confusion_matrix[:, i])
            for i in range(confusion_matrix.shape[0])
        ]
    )
    precision = np.array(
        [
            confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])
            for i in range(confusion_matrix.shape[0])
        ]
    )

    recall = np.nan_to_num(recall)
    precision = np.nan_to_num(precision)

    if return_ave:
        recall = np.mean(recall)
        precision = np.mean(precision)

    f1_score_ave = np.divide(
        2 * precision * recall,
        (precision + recall),
    )

    f1_score_ave = np.nan_to_num(f1_score_ave)

    return (
        np.float64(precision),
        np.float64(recall),
        np.float64(f1_score_ave),
    )


def validate_metrics(
    current_metrics: Dict[str, Union[float, Tensor]],
    best_metrics: Dict[str, Union[float, Tensor]],
    cfg_path: str,
    model: Module,
    epoch: int,
    optimizer: Optimizer,
    current_loss: float,
) -> None:
    """Validate metrics and save checkpoint if best

    Args:
        current_metrics (Dict[str, Union[float, Tensor]]): Current metrics
        best_metrics (Dict[str, Union[float, Tensor]]): Best metrics
        cfg_path (str): Path to config file
        model (Module): Model to save
        epoch (int): Epoch number
        optimizer (Optimizer): Optimizer to save
        current_loss (float): Current train loss
    """
    cfg = get_cfg_from_file(cfg_path)
    cfg_name = os.path.basename(cfg_path).split(".")[0]
    for metric_str, value in current_metrics.items():
        if "confusion_matrix" in metric_str:
            continue
        if metric_str not in best_metrics:
            update = True
        else:
            best_metric = best_metrics[metric_str]
            if metric_str in ["val_loss"]:
                update = value < best_metric
            else:
                update = value > best_metric

        if update:
            best_metrics[metric_str] = value
            save_path = os.path.join(
                cfg.TRAIN.WEIGHTS_FOLDER,
                f"cfg_{cfg_name}_best_{metric_str}.pth",
            )
            logger.info(f"Saving checkpoint for the best {metric_str}: {value:.4f}")
            save_checkpoint(model, epoch, optimizer, current_loss, cfg_path, save_path)
