import glob
import os
from typing import List
from shutil import rmtree

from torch import Tensor
from torch.utils.data import DataLoader

from utils.raster_utils import is_cropped, crop_raster
from utils.visualization_utils import (
    generate_save_alphablend,
    generate_save_alphablended_raster,
    generate_save_raster,
    generate_save_raw_raster,
)
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


def get_path_for_output(output_type, destination, name, dataloader):
    output_destination = os.path.join(destination, output_type)
    os.makedirs(output_destination, exist_ok=True)
    extension = "tif" if "raster" in output_type else "png"
    if dataloader.dataset.mode == "infer":
        name = os.path.splitext(os.path.split(name)[1])[0]
        alphablend_path = os.path.join(output_destination, name + f".{extension}")
    else:
        alphablend_path = get_save_path(
            name, output_destination, output_type, extension
        )

    return alphablend_path


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

    if os.path.isfile(name):
        ref_raster_path = name
    else:
        ref_raster_path = get_raster_filepath(
            dataloader.dataset.dataset_root,
            name,
            dataloader.dataset.input_sensor_name,
        )

    for output_type in output_types:
        assert output_type in [
            "alphablend",
            "alphablended_raster",
            "raster",
            "raw_raster"
        ], f"Output type {output_type} not supported"
        output_path = get_path_for_output(output_type, destination, name, dataloader)

        if output_type == "alphablend":
            generate_save_alphablend(input_img, mask, mask_config, output_path)
        elif output_type == "alphablended_raster":
            generate_save_alphablended_raster(
                mask,
                input_img,
                mask_config,
                ref_raster_path,
                output_path,
            )
        elif output_type == "raw_raster":
            generate_save_raw_raster(
                input_img,
                ref_raster_path,
                output_path,
            )
        elif output_type == "raster":
            generate_save_raster(mask, mask_config, ref_raster_path, output_path)


def prepare_raster_for_inference(input_raster: str, crop_size: List[int]):
    paths_to_infer = []
    raster_folder, raster_file = os.path.split(input_raster)

    if not is_cropped(input_raster, crop_size):
        paths_to_infer.append(input_raster)
    else:

        raster_name = os.path.splitext(raster_file)[0]
        cropped_rasters_directory = os.path.join(raster_folder, raster_name)

        if os.path.isdir(cropped_rasters_directory):
            rmtree(cropped_rasters_directory)
        os.makedirs(cropped_rasters_directory)

        crop_raster(input_raster, cropped_rasters_directory, crop_size)

        paths_to_infer.extend(glob.glob(f"{cropped_rasters_directory}/*.tif"))

    return paths_to_infer
