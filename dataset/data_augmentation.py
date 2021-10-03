from albumentations import Compose
from albumentations.augmentations.transforms import RandomBrightnessContrast
from albumentations.augmentations.crops.transforms import RandomResizedCrop

from yacs.config import CfgNode


def get_augmentation_transforms(cfg: CfgNode):
    transforms = []
    # Add data augmentation
    if cfg.DATA_AUG.RANDOM_RESIZE_CROP.USE:
        transform = RandomResizedCrop(
            height=cfg.DATASET.SHAPE[0],
            width=cfg.DATASET.SHAPE[1],
            scale=cfg.DATA_AUG.RANDOM_RESIZE_CROP.SCALE,
            ratio=cfg.DATA_AUG.RANDOM_RESIZE_CROP.RATIO,
            p=cfg.DATA_AUG.RANDOM_RESIZE_CROP.PROBABILITY,
        )
        transforms.append(transform)
    if cfg.DATA_AUG.RANDOM_BRIGHTNESS_CONTRAST.USE:
        transform = RandomBrightnessContrast(
            brightness_limit=cfg.DATA_AUG.RANDOM_BRIGHTNESS_CONTRAST.BRIGHTNESS_CHANGE,
            contrast_limit=cfg.DATA_AUG.RANDOM_BRIGHTNESS_CONTRAST.CONTRAST_CHANGE,
            p=cfg.DATA_AUG.RANDOM_BRIGHTNESS_CONTRAST.PROBABILITY,
        )
        transforms.append(transform)

    return Compose(transforms)
