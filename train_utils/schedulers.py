import logging

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from yacs.config import CfgNode

from utils.logger import init_log

logger = logging.getLogger("global")


def get_lr_scheduler(optimizer: Optimizer, cfg: CfgNode, start_epoch: int = 0):
    """Returns LR scheduler module"""

    # Get mode
    if cfg.TRAIN.LOSS.TYPE in ["categorical_crossentropy", "focal_loss"]:
        mode = "min"
    else:
        raise NotImplementedError

    if cfg.TRAIN.SCHEDULER.TYPE == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode,
            factor=cfg.TRAIN.SCHEDULER.FACTOR,
            patience=cfg.TRAIN.SCHEDULER.PATIENCE,
            verbose=True,
        )
    elif cfg.TRAIN.SCHEDULER.TYPE == "StepLR":
        scheduler = StepLR(
            optimizer,
            step_size=cfg.TRAIN.SCHEDULER.PATIENCE,
            gamma=cfg.TRAIN.SCHEDULER.FACTOR,
            last_epoch=start_epoch - 1,
        )
    elif cfg.TRAIN.SCHEDULER.TYPE == "None":
        scheduler = None
    else:
        raise NotImplementedError

    logger.info(f"Used scheduler: {scheduler}")

    return scheduler


def update_scheduler(cfg: CfgNode, scheduler, val_loss: float) -> None:
    """Updates scheduler"""
    if cfg.TRAIN.SCHEDULER.TYPE == "ReduceLROnPlateau":
        scheduler.step(val_loss)
    elif cfg.TRAIN.SCHEDULER.TYPE == "StepLR":
        scheduler.step()
    elif cfg.TRAIN.SCHEDULER.TYPE == "None":
        pass
    else:
        raise NotImplementedError
