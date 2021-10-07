import logging
import os

from numpy import random
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from config.default import CfgNode
from dataset.data_augmentation import get_augmentation_transforms
from dataset.dataset_utils import (
    build_classes_distribution_json,
    get_classes_counts_from_json,
)
from dataset.patch_dataset import PatchDataset
from dataset.transforms import get_transform
from utils.utilities import (
    build_dataset_stats_json_from_cfg,
    get_gpu_count,
)
from utils.io_utils import load_yaml

logger = logging.getLogger("global")


def print_dataloader(dataloader: DataLoader):
    """Returns str of dataloader used by the logger"""
    s = ""
    s += f"Samples count: {len(dataloader.dataset)}\n"
    s += f"Samples list: {dataloader.dataset.dataset_list_path}\n"
    s += f"Batches count: {len(dataloader)}\n"
    s += f"Drop last: {dataloader.drop_last}\n"
    s += f"Transforms: {dataloader.dataset.transforms}\n"
    s += f"Masks config: {load_yaml(dataloader.dataset.cfg.DATASET.MASK.CONFIG)}"
    s += f"Input sensor name: {dataloader.dataset.cfg.DATASET.INPUT.SENSOR}"
    s += f"All channels: {dataloader.dataset.cfg.DATASET.INPUT.CHANNELS}"
    s += f"Used channels: {dataloader.dataset.cfg.DATASET.INPUT.USED_CHANNELS}"
    s += f"Used channels: {dataloader.dataset.cfg.DATASET.MASK.SENSOR}"
    s += "\n"
    return s


def get_dataloader(cfg: CfgNode, samples_list: str) -> DataLoader:
    """Builds and returns a dataloader for the dataset.

    Args:
        cfg (CfgNode): Config object.
        samples_list (str): Either a path to a text file containing the
                                list of samples or one of ["train", "val", "test"].

    Returns:
        DataLoader: [description]
    """

    if not os.path.isfile(cfg.DATASET.INPUT.STATS_FILE):
        build_dataset_stats_json_from_cfg(cfg)

    transform = get_transform(cfg)
    transforms = Compose([transform])
    aug_transforms = get_augmentation_transforms(cfg)

    dataset = PatchDataset(
        cfg, samples_list, transforms=transforms, aug_transforms=aug_transforms
    )

    if samples_list in ["train", "val"]:
        num_workers = cfg.TRAIN.WORKERS
        shuffle = cfg.TRAIN.SHUFFLE
        batch_size = cfg.TRAIN.BATCH_SIZE_PER_DEVICE * get_gpu_count(cfg, samples_list)
        drop_last = True
    else:
        num_workers = cfg.TEST.WORKERS
        shuffle = False
        batch_size = cfg.TEST.BATCH_SIZE_PER_DEVICE * get_gpu_count(cfg, samples_list)
        drop_last = False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        worker_init_fn=random.seed(cfg.TRAIN.SEED),
        drop_last=drop_last,
    )

    logger.info(
        f"\nDataloader used for {samples_list}:\n" + print_dataloader(dataloader) + "\n"
    )

    if not cfg.IS_TEST and samples_list in ["train", "val"]:
        if not os.path.isfile(cfg.DATASET.CLASSES_COUNT_JSON):
            build_classes_distribution_json(cfg, dataloader.dataset.mask_config)
        counts = get_classes_counts_from_json(cfg, samples_list)
        logger.info(f"Train counts: {counts}")

    return dataloader
