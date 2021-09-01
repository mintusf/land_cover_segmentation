import logging
import os

from numpy import random
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from config.default import CfgNode
from dataset.dataset_utils import get_classes_counts_from_df
from dataset.patch_dataset import PatchDataset
from dataset.transforms import get_transform
from utils.utilities import build_dataset_stats_json_from_cfg, get_gpu_count
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


def get_dataloader(cfg: CfgNode, mode: str) -> DataLoader:

    if not os.path.isfile(cfg.DATASET.INPUT.STATS_FILE):
        build_dataset_stats_json_from_cfg(cfg)

    transform = get_transform(cfg)
    transforms = Compose([transform])

    dataset = PatchDataset(cfg, mode, transforms)

    num_workers = cfg.TRAIN.WORKERS
    shuffle = cfg.TRAIN.SHUFFLE

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_DEVICE * get_gpu_count(cfg),
        num_workers=num_workers,
        shuffle=shuffle,
        worker_init_fn=random.seed(cfg.TRAIN.SEED),
        drop_last=True,
    )

    logger.info(
        f"\nDataloader used for {mode}:\n" + print_dataloader(dataloader) + "\n"
    )

    if not cfg.IS_TEST:
        counts = get_classes_counts_from_df(
            dataloader, "/data/seg_data/training_labels.csv"
        )
        logger.info(f"Train counts: {counts}")

    return dataloader
