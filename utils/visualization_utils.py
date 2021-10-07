from typing import Dict, Tuple, Union

import cv2
import numpy as np

import torch
from torch import Tensor
from torchvision.transforms import Compose

from config.default import CfgNode
from dataset.dataset_utils import build_mask
from dataset.transforms import get_transform
from utils.io_utils import load_yaml
from utils.raster_utils import (
    raster_to_tensor,
    raster_to_np,
    convert_np_for_vis,
    np_to_torch,
    np_to_raster,
)
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


def create_alphablend(
    img: np.array,
    mask: np.array,
    alpha: float,
    colors_dict: dict,
    class2label: Union[None, Dict[int, str]] = None,
) -> np.array:
    """A method to create alphablend image

    Args:
        img (np.array): Input image
        mask (np.array): Mask
        alpha (float): Alpha value
        colors_dict (dict): Dictionary matching class id to color
        class2label (dict): Dictionary matching class id to label

    Returns:
        np.array: Alphablend image
    """

    x_pos = 30
    y_pox = 30
    for class_int, color in colors_dict.items():
        class_mask = np.where(mask == class_int, 1, 0)
        img = apply_single_mask(img, class_mask, color, alpha)
        if class_mask.sum() > 100 and class2label is not None:
            cv2.putText(
                img,
                class2label[class_int],
                (x_pos, y_pox),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            y_pox += 30

    return img


def generate_save_alphablend(
    input_img: Tensor,
    mask: Tensor,
    mask_config: dict,
    alphablend_path: str,
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
    class2label = mask_config["class2label"]
    alphablended = create_alphablend(input_img, mask, alpha, colors_dict, class2label)
    alphablended = cv2.cvtColor(alphablended, cv2.COLOR_BGR2RGB)
    cv2.imwrite(alphablend_path, alphablended)


def generate_save_raster(
    mask: Tensor, mask_config: dict, ref_raster: str, raster_path: str
) -> None:
    """Generates and saves raster of a mask

    Args:
        mask (Tensor): Mask tensor.
                       Shape (1, H, W) where pixel value corresponds to class int
        mask_config (dict): Mask config
        ref_raster (str): Path to reference raster, used to get transform and crs
        raster_destination (str): Root path to save raster
    """
    mask = mask.cpu().numpy()
    dummy_image = np.ones((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
    alpha = 1.0
    colors_dict = mask_config["colors"]
    alphablended = create_alphablend(dummy_image, mask, alpha, colors_dict)
    colored_mask = alphablended.transpose(2, 0, 1)
    colored_mask = colored_mask.astype(np.int8)
    np_to_raster(colored_mask, ref_raster, raster_path)


def generate_save_alphablended_raster(
    mask: Tensor,
    input_img: Tensor,
    mask_config: dict,
    ref_raster: str,
    raster_path: str,
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
    np_to_raster(alphablended, ref_raster, raster_path)


def generate_save_raw_raster(
    input_img: Tensor,
    ref_raster: str,
    raster_path: str,
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
    input_img = prepare_tensors_for_vis(input_img, None)
    input_img = input_img.transpose(2, 0, 1)
    np_to_raster(input_img, ref_raster, raster_path)


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
    input_used_channels = cfg.DATASET.INPUT.USED_CHANNELS
    input_tensor = raster_to_tensor(input_raster_path, bands=input_used_channels)

    transform = get_transform(cfg)
    transforms = Compose([transform])
    input_img = transforms({"input": input_tensor})["input"]

    # Get mask
    target_raster_path = get_raster_filepath(
        dataset_root, sample_name, target_sensor_name
    )
    target_np = raster_to_np(target_raster_path)
    mask = build_mask(target_np, mask_config)
    mask = np_to_torch(mask, dtype=torch.long)

    # Create alphablend
    generate_save_alphablend(input_img, mask, mask_config, savepath)


def prepare_tensors_for_vis(
    input_img: Tensor, mask: Union[None, Tensor]
) -> Union[np.array, Tuple[np.array, np.array]]:
    """Prepares input and mask for visualization

    Args:
        input_img (Tensor): Input img tensor
        mask (Tensor): Predicted mask tensor

    Returns:
        Tuple[np.array, np.array]: Input and mask for visualization
    """
    input_img = input_img.cpu().numpy()
    input_img = input_img[(2, 1, 0), :, :]
    input_img = convert_np_for_vis(input_img)

    if mask is None:
        return input_img
    else:
        mask = mask.cpu().numpy()
        return input_img, mask
