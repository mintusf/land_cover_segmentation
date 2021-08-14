from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from yacs.config import CfgNode


def get_lr_scheduler(optimizer: Optimizer, cfg: CfgNode):
    """Returns LR scheduler module"""

    # Get mode
    if cfg.TRAIN.LOSS == "categorical_crossentropy":
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

    return scheduler