import os
from typing import Dict

import torch
from torch import tensor
from torch.utils.data import Dataset
from yacs.config import CfgNode

from dataset.dataset_utils import build_mask
from utils.io_utils import get_lines_from_txt, load_yaml
from utils.raster_utils import raster_to_tensor, raster_to_np, np_to_torch
from utils.utilities import get_raster_filepath


class PatchDataset(Dataset):
    def __init__(self, cfg: CfgNode, mode: str, infer_list: str, transforms=None):
        """Patch Dataset initialization

        Args:
            cfg (CfgNode): Config
        """
        self.cfg = cfg

        self.dataset_root = cfg.DATASET.ROOT
        self.mask_config = load_yaml(cfg.DATASET.MASK.CONFIG)
        self.input_sensor_name = cfg.DATASET.INPUT.SENSOR
        self.channels_list = cfg.DATASET.INPUT.CHANNELS
        self.input_used_channels = cfg.DATASET.INPUT.USED_CHANNELS
        self.target_sensor_name = cfg.DATASET.MASK.SENSOR

        if mode == "train":
            self.dataset_list = get_lines_from_txt(cfg.DATASET.LIST_TRAIN)
        elif mode == "val":
            self.dataset_list = get_lines_from_txt(cfg.DATASET.LIST_VAL)
        elif mode == "test":
            self.dataset_list = get_lines_from_txt(cfg.DATASET.LIST_TEST)
        elif mode == "infer":
            self.dataset_list = get_lines_from_txt(infer_list)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.transforms = transforms
        self.device = cfg.TRAIN.DEVICE if mode in ["train", "val"] else cfg.TEST.DEVICE

        self.mode = mode

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
        # Get sample name
        sample_name = self.dataset_list[index]

        # Get input tensor
        input_raster_path = get_raster_filepath(
            self.dataset_root, sample_name, self.input_sensor_name
        )
        input_tensor = raster_to_tensor(
            input_raster_path, bands=self.input_used_channels
        )

        if self.mode != "infer":
            # Get target tensor
            target_raster_path = get_raster_filepath(
                self.dataset_root, sample_name, self.target_sensor_name
            )
            target_np = raster_to_np(target_raster_path)
            transformed_mask = build_mask(target_np, self.mask_config)
            target_tensor = np_to_torch(transformed_mask, dtype=torch.long)

        if "cuda" in self.device:
            if "all" in self.device:
                device = 0
            else:
                devices = self.device.split(":")[1].split(",")
                device = devices[0]
            input_tensor = input_tensor.to(torch.device(f"cuda:{device}")).float()
            if self.mode != "infer":
                target_tensor = target_tensor.to(torch.device(f"cuda:{device}"))
        elif "cpu" in self.device:
            input_tensor = input_tensor.cpu().float()
            if self.mode != "infer":
                target_tensor = target_tensor.cpu().long()
        else:
            raise NotImplementedError

        # Return sample
        if self.mode != "infer":
            sample = {
                "input": input_tensor,
                "target": target_tensor,
                "name": sample_name,
            }
        else:
            sample = {"input": input_tensor, "name": sample_name}

        # Tranform
        if self.transforms:
            sample = self.transforms(sample)

        return sample
