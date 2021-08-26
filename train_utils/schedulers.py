import logging

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from yacs.config import CfgNode

from utils.logger import init_log

logger = logging.getLogger("global")


def get_lr_scheduler(optimizer: Optimizer, cfg: CfgNode):
    """Returns LR scheduler module"""

    # Get mode
    if cfg.TRAIN.LOSS in ["categorical_crossentropy", "focal_loss"]:
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
    elif cfg.TRAIN.SCHEDULER.TYPE == "None":
        scheduler = None
    else:
        raise NotImplementedError

    logger.info(f"Used scheduler: {scheduler}")

    return scheduler
