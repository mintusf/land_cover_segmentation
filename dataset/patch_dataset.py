import os
from typing import Dict

import torch
from torch import tensor
from torch.utils.data import Dataset
from yacs.config import CfgNode

from dataset.dataset_utils import build_mask
from utils.io_utils import get_lines_from_txt, load_yaml
from utils.raster_utils import raster_to_np, np_to_torch
from utils.utilities import get_raster_filepath


class PatchDataset(Dataset):
    def __init__(
        self, cfg: CfgNode, samples_list: str, transforms=None, aug_transforms=None
    ):
        """Patch Dataset initialization

        Args:
            cfg (CfgNode): Config
            samples_list (str): Either a path to a text file containing the
                                list of samples or one of ["train", "val", "test"].
                                If a path, Dataset is used in inference mode and
                                only input is generated.
            transforms (callable, optional): Optional transform to be applied
            aug_transforms (callable, optional): Optional data augmentation transforms
                                                 to be applied
        """
        self.cfg = cfg

        self.dataset_root = cfg.DATASET.ROOT
        self.mask_config = load_yaml(cfg.DATASET.MASK.CONFIG)
        self.input_sensor_name = cfg.DATASET.INPUT.SENSOR
        self.channels_list = cfg.DATASET.INPUT.CHANNELS
        self.input_used_channels = cfg.DATASET.INPUT.USED_CHANNELS
        self.target_sensor_name = cfg.DATASET.MASK.SENSOR

        if samples_list == "train":
            self.dataset_list_path = cfg.DATASET.LIST_TRAIN
        elif samples_list == "val":
            self.dataset_list_path = cfg.DATASET.LIST_VAL
        elif samples_list == "test":
            self.dataset_list_path = cfg.DATASET.LIST_TEST
        else:
            assert os.path.isfile(
                samples_list
            ), f"Invalid samples list path {samples_list}"
            self.dataset_list_path = samples_list
            samples_list = "infer"

        self.mode = samples_list

        self.dataset_list = get_lines_from_txt(self.dataset_list_path, shuffle=True)

        self.transforms = transforms
        self.aug_transforms = aug_transforms

        self.device = (
            cfg.TRAIN.DEVICE if self.mode in ["train", "val"] else cfg.TEST.DEVICE
        )

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

        # Get input numpy array
        if os.path.isfile(sample_name):
            input_raster_path = sample_name
        else:
            input_raster_path = get_raster_filepath(
                self.dataset_root, sample_name, self.input_sensor_name
            )
        input_np = raster_to_np(input_raster_path, bands=self.input_used_channels)

        # Get target numpy array
        if self.mode != "infer":
            # Get target tensor
            target_raster_path = get_raster_filepath(
                self.dataset_root, sample_name, self.target_sensor_name
            )
            target_np = raster_to_np(target_raster_path)
            transformed_mask = build_mask(target_np, self.mask_config)

        # Apply data augmentation
        if self.mode == "train":
            if self.aug_transforms is not None:
                input_np = input_np.transpose(1, 2, 0)
                augmented = self.aug_transforms(image=input_np, mask=transformed_mask)
                augmented["image"] = augmented["image"].transpose(2, 0, 1)
                input_np = augmented["image"]
                transformed_mask = augmented["mask"]

        if "cuda" in self.device:
            if "all" in self.device:
                device = 0
            else:
                devices = self.device.split(":")[1].split(",")
                device = devices[0]
            device = torch.device(f"cuda:{device}")
        elif "cpu" in self.device:
            device = torch.device("cpu")
        else:
            raise NotImplementedError

        input_tensor = np_to_torch(input_np)
        input_tensor = input_tensor.to(device).float()

        sample = {"input": input_tensor, "name": sample_name}

        if self.mode != "infer":
            target_tensor = np_to_torch(transformed_mask, dtype=torch.long)
            target_tensor = target_tensor.to(device).long()
            sample["target"] = target_tensor

        # Transform
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
