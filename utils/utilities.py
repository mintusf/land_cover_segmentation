import json
from typing import List
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from yacs.config import CfgNode

from utils.raster_utils import get_stats
from utils.io_utils import get_lines_from_txt


def split_sample_name(sample_name: str) -> str:
    """Split sample name into ROI folder name, area, and subgrid ID."""
    parts = sample_name.split("_")
    roi_folder_name = "_".join(parts[:2])
    area = parts[2]
    subgrid_id = parts[3]
    return roi_folder_name, area, subgrid_id


def get_area_foldername(sensor: str, area: str) -> str:
    """Get area foldername given sensor and area"""
    return f"{sensor}_{area}"


def get_raster_filepath(rootdir: str, sample_name: str, sensor: str) -> str:
    """Get raster filepath given rootdir, sample name, and sensor
    Args:
        rootdir (str): root directory of the dataset
        sample_name (str): sample name, e.g "ROIs2017_winter_27_p36"
        sensor (str): sensor name

    Returns:
        str: raster filepath
    """
    roi_folder_name, area, subgrid_id = split_sample_name(sample_name)
    folder = os.path.join(rootdir, roi_folder_name, get_area_foldername(sensor, area))
    filename = f"{roi_folder_name}_{sensor}_{area}_{subgrid_id}.tif"
    return os.path.join(folder, filename)


def build_dataset_stats_json(
    dataset_list: List[str],
    dataset_root: str,
    input_sensor_name: str,
    channels_list: List[int],
    savepath: str,
):
    """Builds stats json for a dataset.

    Args:
        dataset_list (list): List of dataset.
        dataset_root (str): Root directory of the dataset.
        input_sensor_name (str): Name of the input sensor.
        channels_list (list): List of channels.
        savepath (str): Path to save the json.
    """
    means = []
    stds = []
    filepaths = [
        get_raster_filepath(dataset_root, sample_name, input_sensor_name)
        for sample_name in dataset_list
    ]

    for file in filepaths:
        image_means, image_stds = get_stats(file)
        means.append(image_means)
        stds.append(image_stds)

    means = np.array(means)
    stds = np.array(stds)

    # Since all images have same amount of pixels,
    # mean of combination is mean of means
    global_mean = np.mean(means, axis=0)

    # Calculate std of combination
    _N = stds.shape[0]
    std_squared_sum = np.sum(stds ** 2, axis=0)
    means_difference_squared_sum = np.sum((means - global_mean) ** 2, axis=0)
    global_std = ((std_squared_sum + means_difference_squared_sum) / (_N)) ** 0.5

    means_dict = {band: mean.item() for band, mean in zip(channels_list, global_mean)}
    stds_dict = {band: std.item() for band, std in zip(channels_list, global_std)}

    with open(savepath, "w") as f:
        json.dump({"means": means_dict, "stds": stds_dict}, f)


def build_dataset_stats_json_from_cfg(cfg: CfgNode) -> None:
    """Builds stats json for a dataset given config.

    Args:
        cfg (CfgNode): A Yacs CfgNode object.
    """
    dataset_list = get_lines_from_txt(cfg.DATASET.LIST_TRAIN)
    build_dataset_stats_json(
        dataset_list,
        cfg.DATASET.ROOT,
        cfg.DATASET.INPUT.SENSOR,
        cfg.DATASET.INPUT.CHANNELS,
        cfg.DATASET.INPUT.STATS_FILE,
    )


def get_sample_name(filename: str) -> str:
    """Get sample name from filename."""
    split = filename.split("_")
    return "_".join(split[:2] + split[3:])


def get_gpu_count(cfg: CfgNode) -> int:
    """Returns used GPUs count given config"""
    if "cpu" in cfg.TRAIN.DEVICE:
        devices = 1
    elif "all" in cfg.TRAIN.DEVICE:
        devices = torch.cuda.device_count()
    else:
        devices = len(cfg.TRAIN.DEVICE.split(":")[1].split(","))
    return devices


def is_intersection_empty(dataloader1: DataLoader, dataloader2: DataLoader) -> bool:
    """Checks if no sample in both train and checked dataloader"""
    samples1 = set(dataloader1.dataset.dataset_list)
    samples2 = set(dataloader2.dataset.dataset_list)
    return samples1.isdisjoint(samples2)
