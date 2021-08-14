import os

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from config.default import CfgNode
from dataset.patch_dataset import PatchDataset
from dataset.transforms import get_transform
from utils.utils import build_dataset_stats_json_from_cfg


def get_dataloader(cfg: CfgNode, mode: str) -> DataLoader:

    if not os.path.isfile(cfg.DATASET.INPUT.STATS_FILE):
        build_dataset_stats_json_from_cfg(cfg)

    transform = get_transform(cfg)
    transforms = Compose([transform])

    dataset = PatchDataset(cfg, mode, transforms)

    batch_size = cfg.TRAIN.BATCH_SIZE
    num_workers = cfg.TRAIN.WORKERS
    shuffle = cfg.TRAIN.SHUFFLE

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
    )

    return dataloader
