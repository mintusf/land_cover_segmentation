import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from config.default import CfgNode, get_cfg_from_file
from dataset.dataset_utils import build_mask
from utils.io_utils import load_yaml, get_lines_from_txt
from utils.raster_utils import raster_to_np
from utils.utils import get_raster_filepath


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
    classes_mask = mask.unique()
    classes = config["class2label"].keys().tolist()
    assert set(classes_mask) - set(classes) == {}

    alpha = config["alpha"]
    colors_dict = config["colors"]

    for class_int, color in colors_dict.items():
        img = apply_single_mask(img, mask, color, alpha)

    return img


def vis_sample(sample_name: str, cfg: CfgNode, savepath: str) -> None:
    # Parse config
    dataset_root = cfg.DATASET.ROOT
    mask_config = load_yaml(cfg.DATASET.MASK.CONFIG)
    input_sensor_name = cfg.DATASET.INPUT.SENSOR
    channels_list = cfg.DATASET.INPUT.CHANNELS
    target_sensor_name = cfg.DATASET.MASK.SENSOR

    # Get image
    input_raster_path = get_raster_filepath(
        dataset_root, sample_name, input_sensor_name
    )
    img = raster_to_np(input_raster_path, channels_list)

    # Get mask
    target_raster_path = get_raster_filepath(
        dataset_root, sample_name, target_sensor_name
    )
    target_np = raster_to_np(target_raster_path)
    mask = build_mask(target_np, mask_config)

    # Create alphablend
    alphablend = create_alphablend(img, mask, mask_config)

    # Save
    cv2.imwrite(savepath, alphablend)


cfg_path = "config/firstrun.yml"
cfg = get_cfg_from_file(cfg_path)
samples = get_lines_from_txt(cfg.DATASET.TRAIN_LIST)
i = 0
for sample in samples:
    if i > 50:
        break
    i += 1
    vis_sample(sample, cfg, f"vis/{sample}.png")
