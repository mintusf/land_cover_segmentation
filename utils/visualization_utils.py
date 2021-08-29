import os
import sys

import cv2
import numpy as np

from config.default import CfgNode
from dataset.dataset_utils import build_mask
from utils.io_utils import load_yaml, load_json
from utils.raster_utils import convert_raster_for_vis, raster_to_np
from utils.utilities import get_raster_filepath


def apply_single_mask(
    image: np.array, mask: np.array, color: tuple, alpha: float = 0.6
) -> np.array:
    """A method to generate visualization of masks
    Args:
        image (np.array): Input image
        mask (np.array): Mask
        color (tuple): Color of mask (R, G, B)
        alpha (float, optional): Non-transparency of mask. Defaults to 0.6.
    Returns:
        np.array: Image with mask visualization
    """
    out = image.copy()
    for c in range(3):
        out[:, :, c] = np.where(
            mask != 0, image[:, :, c] * (1 - alpha) + alpha * color[c], image[:, :, c]
        )
    return out


def create_alphablend(img: np.array, mask: np.array, config: dict) -> np.array:
    """A method to create alphablend image

    Args:
        img (np.array): Input image
        mask (np.array): Mask
        config (dict): Mask config

    Returns:
        np.array: Alphablend image
    """
    classes_mask = np.unique(mask)
    classes = list(config["class2label"].keys())
    assert not (set(classes_mask) - set(classes))

    alpha = config["alpha"]
    colors_dict = config["colors"]

    for class_int, color in colors_dict.items():
        class_mask = np.where(mask == class_int, 1, 0)
        img = apply_single_mask(img, class_mask, color, alpha)

    return img


def vis_sample(sample_name: str, cfg: CfgNode, savepath: str) -> None:
    """A method to visualize a sample

    Args:
        sample_name (str): Sample name
        cfg (CfgNode): Config
        savepath (str): Save path
    """
    # Parse config
    dataset_root = cfg.DATASET.ROOT
    mask_config = load_yaml(cfg.DATASET.MASK.CONFIG)
    input_sensor_name = cfg.DATASET.INPUT.SENSOR
    target_sensor_name = cfg.DATASET.MASK.SENSOR

    # Get image
    input_raster_path = get_raster_filepath(
        dataset_root, sample_name, input_sensor_name
    )
    img = convert_raster_for_vis(input_raster_path)

    # Get mask
    target_raster_path = get_raster_filepath(
        dataset_root, sample_name, target_sensor_name
    )
    target_np = raster_to_np(target_raster_path)
    mask = build_mask(target_np, mask_config)

    # Create alphablend
    alphablend = create_alphablend(img, mask, mask_config)

    # Save
    alphablend = cv2.cvtColor(alphablend, cv2.COLOR_RGB2BGR)
    cv2.imwrite(savepath, alphablend)
