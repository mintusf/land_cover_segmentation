from typing import List

import os

import cv2
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader

from utils.raster_utils import np_to_raster
from utils.visualization_utils import create_alphablend, prepare_tensors_for_vis
from utils.utilities import split_sample_name, get_raster_filepath


def get_save_path(
    name: str, destination: str, suffix: str, extention: str = "png"
) -> str:
    """Returns a path for an output sample.
        Creates directory if doesn't exist

    Args:
        name (str): Sample name
        destination (str): Root directory
        suffix (str): Suffix for file
    Returns:
        str: Save path
    """
    roi_folder, area, _ = split_sample_name(name)
    alphablend_folder = os.path.join(destination, roi_folder, area)
    if not os.path.isdir(alphablend_folder):
        os.makedirs(alphablend_folder)
    alphablend_path = os.path.join(alphablend_folder, f"{name}_{suffix}.{extention}")

    return alphablend_path


def generate_save_alphablend(
    input_img: Tensor,
    mask: Tensor,
    name: str,
    mask_config: dict,
    alphablend_destination: str,
):
    """Generates and saves alphablend

    Args:
        input_img (Tensor): Input img tensor
        mask (Tensor): Predicted mask tensor
        name (str): Sample name
        mask_config (dict): Mask config
        alphablend_destination (str): Root path to save alphablend
    """
    input_img, mask = prepare_tensors_for_vis(input_img, mask)
    alpha = mask_config["alpha"]
    colors_dict = mask_config["colors"]
    alphablended = create_alphablend(input_img, mask, alpha, colors_dict)
    alphablend_path = get_save_path(name, alphablend_destination, "alphablend")
    cv2.imwrite(alphablend_path, alphablended)


def generate_save_raster(
    mask: Tensor, name: str, mask_config: dict, ref_raster: str, raster_destination: str
) -> None:
    """Generates and saves raster of a mask

    Args:
        mask (Tensor): Mask tensor.
                       Shape (1, H, W) where pixel value corresponds to class int
        name (str): Sample name
        mask_config (dict): Mask config
        ref_raster (str): Path to reference raster, used to get transform and crs
        raster_destination (str): Root path to save raster
    """
    mask = mask.cpu().numpy()
    dummy_image = np.ones((mask.shape[0], mask.shape[1], 3))
    alpha = 1.0
    colors_dict = mask_config["colors"]
    alphablended = create_alphablend(dummy_image, mask, alpha, colors_dict)
    colored_mask = alphablended.transpose(2, 0, 1)
    raster_path = get_save_path(name, raster_destination, "raster", extention="tif")
    np_to_raster(colored_mask, ref_raster, raster_path)


def generate_save_alphablended_raster(
    mask: Tensor,
    input_img: Tensor,
    name: str,
    mask_config: dict,
    ref_raster: str,
    raster_destination: str,
) -> None:
    """Generates and saves raster of a mask

    Args:
        mask (Tensor): Mask tensor.
                       Shape (1, H, W) where pixel value corresponds to class int
        input_img (Tensor): Input img tensor
        name (str): Sample name
        mask_config (dict): Mask config
        ref_raster (str): Path to reference raster, used to get transform and crs
        raster_destination (str): Root path to save raster
    """
    input_img, mask = prepare_tensors_for_vis(input_img, mask)
    alpha = mask_config["alpha"]
    colors_dict = mask_config["colors"]
    alphablended = create_alphablend(input_img, mask, alpha, colors_dict)
    alphablended = alphablended.transpose(2, 0, 1)
    raster_path = get_save_path(
        name, raster_destination, "alphablended_raster", extention="tif"
    )
    np_to_raster(alphablended, ref_raster, raster_path)


def generate_outputs(
    output_types: List[str],
    destination: str,
    input_img: Tensor,
    mask: Tensor,
    name: str,
    mask_config: dict,
    dataloader: DataLoader,
) -> None:
    """Generates and saves output images in formats specified by `output_types`

    Args:
        output_types (List[str]): List of output types.
                                  Currently supported:
                                  * alphablended (png alphablend)
                                  * alphablended_raster (tif alphablend)
                                  * raster (tif mask)
        destination (str): Root path to save outputs
        input_img (Tensor): Input img tensor
        mask (Tensor): Predicted mask tensor
        name (str): Sample name
        mask_config (dict): Mask config
        dataloader (DataLoader): Dataloader for samples
    """
    if "alphablend" in output_types:
        alphablend_destination = os.path.join(destination, "alphablend")
        generate_save_alphablend(
            input_img, mask, name, mask_config, alphablend_destination
        )

    ref_raster_path = get_raster_filepath(
        dataloader.dataset.dataset_root,
        name,
        dataloader.dataset.input_sensor_name,
    )

    if "raster" in output_types:
        raster_destination = os.path.join(destination, "raster")
        generate_save_raster(
            mask, name, mask_config, ref_raster_path, raster_destination
        )
    if "alphablended_raster" in output_types:
        raster_destination = os.path.join(destination, "alphablended_raster")

        generate_save_alphablended_raster(
            mask,
            input_img,
            name,
            mask_config,
            ref_raster_path,
            raster_destination,
        )
