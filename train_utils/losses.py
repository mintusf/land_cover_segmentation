import logging
import os
from typing import Optional, List, Dict

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from config.default import CfgNode
from dataset import (
    build_classes_distribution_json,
    get_classes_counts_from_json,
)
from utils.io_utils import load_yaml

logger = logging.getLogger("global")


def get_loss(cfg):

    # Get weights
    if cfg.TRAIN.LOSS.USE_WEIGHTS:
        if cfg.TRAIN.LOSS.TYPE not in ["categorical_crossentropy"]:
            logger.info(f"Loss {cfg.TRAIN.LOSS.TYPE} does not support weights")
            weights = None
        else:
            weights = get_class_weights(cfg)

            if "cuda" in cfg.TRAIN.DEVICE:
                if "all" in cfg.TRAIN.DEVICE:
                    device = 0
                else:
                    devices = cfg.TRAIN.DEVICE.split(":")[1].split(",")
                    device = devices[0]
                device = torch.device(f"cuda:{device}")
            elif "cpu" in cfg.TRAIN.DEVICE:
                device = torch.device("cpu")

            weights = weights.to(device)

            logger.info(f"Used weights: {weights}")
    else:
        weights = None

    if cfg.TRAIN.LOSS.TYPE == "categorical_crossentropy":
        loss = CrossEntropyLoss(weight=weights)
    elif cfg.TRAIN.LOSS.TYPE == "focal_loss":
        loss = FocalLoss(alpha=1.0)
    else:
        raise NotImplementedError(f"Loss {cfg.TRAIN.LOSS.TYPE} is not implemented")

    logger.info(f"Used loss: {loss}")

    return loss


# Focal loss is based on:
# https://github.com/zhezh/focalloss/blob/master/focalloss.py
# https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py


def one_hot(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.
    Args:
        labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.
    Returns:
        the labels in one hot tensor of shape :math:`(N, C, *)`,
    """
    if not isinstance(labels, torch.Tensor):
        raise TypeError(
            "Input labels type is not a torch.Tensor. Got {}".format(type(labels))
        )

    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}".format(labels.dtype)
        )

    if num_classes < 1:
        raise ValueError(
            "The number of classes must be bigger than one."
            " Got: {}".format(num_classes)
        )

    shape = labels.shape
    one_hot = torch.zeros(
        (shape[0], num_classes) + shape[1:], device=device, dtype=dtype
    )

    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    gamma: float = 2.0,
    reduction: str = "none",
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Criterion that computes Focal loss.
    According to :cite:`lin2018focal`, the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    Where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)`
                where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.
    Return:
        the computed loss.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError(
            "Invalid input shape, we expect BxCx*. Got: {}".format(input.shape)
        )

    if input.size(0) != target.size(0):
        raise ValueError(
            "Expected input batch_size ({}) to match target batch_size ({}).".format(
                input.size(0), target.size(0)
            )
        )

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(
            "Expected target size {}, got {}".format(out_size, target.size())
        )

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}".format(
                input.device, target.device
            )
        )

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1) + eps

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(
        target, num_classes=input.shape[1], device=input.device, dtype=input.dtype
    )

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1.0, gamma)

    focal = -alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == "none":
        loss = loss_tmp
    elif reduction == "mean":
        loss = torch.mean(loss_tmp)
    elif reduction == "sum":
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}".format(reduction))
    return loss


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.
    According to :cite:`lin2018focal`, the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    Where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.
    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    """

    def __init__(
        self,
        alpha: float,
        gamma: float = 2.0,
        reduction: str = "mean",
        eps: float = 1e-8,
    ) -> None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(
            input, target, self.alpha, self.gamma, self.reduction, self.eps
        )


def get_class_weights(cfg: CfgNode) -> Tensor:
    """Returns class weights for a given dataset.

    Args:
        cfg (CfgNode): The config node.
        mask_config (dict): The mask config.

    Returns:
        Tensor: Tensor with classes' weights.
    """
    mask_config = load_yaml(cfg.DATASET.MASK.CONFIG)
    classes_count = len(mask_config["class2label"])
    class_importance = mask_config["class_importance"]
    if not os.path.isfile(cfg.DATASET.CLASSES_COUNT_JSON):
        build_classes_distribution_json(cfg, mask_config)
    counts = get_classes_counts_from_json(cfg, "train")
    print(counts)

    weights = [
        (1 / ((counts[str(i)] + 10e-6))) * class_importance[i]
        for i in range(classes_count)
    ]

    return Tensor(weights)
