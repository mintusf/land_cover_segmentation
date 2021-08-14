from torch.optim import Adam


def get_optimizer(model, cfg):
    if cfg.TRAIN.OPTIMIZER == "adam":
        optimizer = Adam(
            model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY
        )
    else:
        raise NotImplementedError

    return optimizer
