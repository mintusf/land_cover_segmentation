from torch.nn import CrossEntropyLoss


def get_loss(cfg):
    if cfg.TRAIN.LOSS == "categorical_crossentropy":
        return CrossEntropyLoss()
