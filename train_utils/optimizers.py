from torch.optim import Adam


def get_optimizer(model, cfg):
    if cfg.TRAIN.LOSS == "adam":
        optimizer = Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WD)
    else:
        raise NotImplementedError

    return optimizer