from albumentations import Compose
from albumentations.augmentations.transforms import HorizontalFlip, VerticalFlip
from albumentations.augmentations.crops.transforms import RandomResizedCrop

from yacs.config import CfgNode


def get_augmentation_transforms(cfg: CfgNode) -> Compose:
    """Returns the augmentation transforms for the dataset.

    Args:
        cfg (CfgNode): The configuration node.

    Returns:
        Compose: The augmentation transforms.
    """
    transforms = []
    # Add data augmentation
    if cfg.TRAIN.DATA_AUG.RANDOM_RESIZE_CROP.USE:
        transform = RandomResizedCrop(
            height=cfg.DATASET.SHAPE[0],
            width=cfg.DATASET.SHAPE[1],
            scale=cfg.TRAIN.DATA_AUG.RANDOM_RESIZE_CROP.SCALE,
            ratio=cfg.TRAIN.DATA_AUG.RANDOM_RESIZE_CROP.RATIO,
            p=cfg.TRAIN.DATA_AUG.RANDOM_RESIZE_CROP.PROBABILITY,
        )
        transforms.append(transform)
    if cfg.TRAIN.DATA_AUG.HORIZONTAL_FLIP.USE:
        transform = HorizontalFlip(
            p=cfg.TRAIN.DATA_AUG.HORIZONTAL_FLIP.PROBABILITY,
        )
        transforms.append(transform)
    if cfg.TRAIN.DATA_AUG.VERTICAL_FLIP.USE:
        transform = VerticalFlip(
            p=cfg.TRAIN.DATA_AUG.VERTICAL_FLIP.PROBABILITY,
        )
        transforms.append(transform)

    return Compose(transforms)
