from typing import Dict

import torch
from torch.utils.data import Dataset
from yacs.config import CfgNode

from utils.io_utils import get_lines_from_txt


class PatchDataset(Dataset):
    def __init__(self, cfg: CfgNode):
        """Patch Dataset initialization

        Args:
            cfg (CfgNode): Config
        """
        self.cfg = cfg
        self.dataset_list = get_lines_from_txt(cfg.DATASET.LIST)

    def __len__(self) -> int:
        """Get length of dataset

        Returns:
            length (int): Length of dataset
        """
        return len(self.dataset_list)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get single sample given index

        Args:
            index (int): Index

        Returns:
            sample (Dict[str, torch.Tensor]): Sample, including:
                                              * input image
                                              * target mask
        """
        raise NotImplementedError
