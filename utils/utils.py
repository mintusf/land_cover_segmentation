import json
from typing import List
import os

import numpy as np
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
    means = np.zeros((len(channels_list)))
    stds = np.zeros((len(channels_list)))
    filepaths = [
        get_raster_filepath(dataset_root, sample_name, input_sensor_name)
        for sample_name in dataset_list
    ]
    for file in filepaths:
        image_means, image_stds = get_stats(file)
        means += image_means
        stds += image_stds

    means = means / len(filepaths)
    stds = stds / len(filepaths)

    means_dict = {band: mean for band, mean in zip(channels_list, means)}
    stds_dict = {band: std for band, std in zip(channels_list, stds)}

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
    return "_".join(split[:3] + split[4:])
