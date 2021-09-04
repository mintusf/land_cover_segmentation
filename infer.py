import argparse
from typing import List
import os

import cv2
import torch
from torch.nn import Module

from config.default import get_cfg_from_file
from dataset import get_dataloader
from models import get_model
from train_utils import load_checkpoint
from utils.raster_utils import convert_np_for_vis
from utils.visualization_utils import create_alphablend
from utils.utilities import split_sample_name, get_gpu_count


def parser():
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="Train the model")

    parser.add_argument(
        "--cfg",
        help="Path to the config file defining testing",
        type=str,
        default="/data/land_cover_tracking/config/weighted_loss.yml",
    )

    parser.add_argument(
        "--checkpoint",
        help="Path to the config file",
        type=str,
        default="/data/land_cover_tracking/weights/cfg_weighted_loss_best_f1.pth",
    )

    parser.add_argument(
        "--samples_list",
        help="Path to the list of samples for inference",
        type=str,
        default="/data/land_cover_tracking/config/dataset/lists/test copy.txt",
    )

    parser.add_argument(
        "--destination",
        help="Path for saving results",
        type=str,
        default="/data/seg_data/masks",
    )

    # TODO Add raster generation support
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=["alphablend"],
        help="What kind of outputs to generate",
    )

    return parser.parse_args()


def rename_ordered_dict_from_parallel(ordered_dict):
    old_keys = list(ordered_dict.keys())
    for key in old_keys:
        key_new = key.replace("module.", "")
        ordered_dict[key_new] = ordered_dict.pop(key)

    return ordered_dict


def rename_ordered_dict_to_parallel(ordered_dict):
    old_keys = list(ordered_dict.keys())
    for key in old_keys:
        key_new = "module." + key
        ordered_dict[key_new] = ordered_dict.pop(key)

    return ordered_dict


def get_alphablend_path(name, alphablend_destination):
    roi_folder, area, _ = split_sample_name(name)
    alphablend_folder = os.path.join(alphablend_destination, roi_folder, area)
    if not os.path.isdir(alphablend_folder):
        os.makedirs(alphablend_folder)
    alphablend_path = os.path.join(alphablend_folder, f"{name}_alphablend.png")

    return alphablend_path


def prepare_tensors_for_vis(input_img, mask):
    input_img = input_img.cpu().numpy()
    input_img = input_img[(1, 2, 3), :, :]
    input_img = convert_np_for_vis(input_img)

    mask = mask.cpu().numpy()
    return input_img, mask


def generate_save_alphablend(
    input_img, mask, name, mask_config, alphablend_destination
):
    input_img, mask = prepare_tensors_for_vis(input_img, mask)
    alphablended = create_alphablend(input_img, mask, mask_config)
    alphablend_path = get_alphablend_path(name, alphablend_destination)
    cv2.imwrite(alphablend_path, alphablended)

    return alphablended


def infer(
    model: Module,
    dataloader: dict,
    output_types: List[str],
    alphablend_destination,
) -> dict:
    """Evaluates test dataset and saves predictions if needed

    Args:
        model (Module): [description]
        dataloader (dict): [description]
        output_types (List[str]): [description]
        alphablend_destination ([type]): [description]

    Returns:
        dict: [description]
    """
    with torch.no_grad():
        model.eval()
        mask_config = dataloader.dataset.mask_config
        for batch in dataloader:
            inputs, names = batch["input"], batch["name"]

            # Forward propagation
            outputs = model(inputs)["out"]

            masks = torch.argmax(outputs, dim=1)

            for input_img, mask, name in zip(inputs, masks, names):

                if "alphablend" in output_types:
                    generate_save_alphablend(
                        input_img, mask, name, mask_config, alphablend_destination
                    )


def run_infer(
    cfg_path: str,
    checkpoint: str,
    samples_list: str,
    destination: str,
    output_types: List[str],
):

    # Build the model
    cfg = get_cfg_from_file(cfg_path)
    device = cfg.TEST.DEVICE

    if cfg.TEST.WORKERS > 0:
        torch.multiprocessing.set_start_method("spawn", force=True)

    _, weights, _, _, _ = load_checkpoint(checkpoint, device)

    model = get_model(cfg, device)
    if get_gpu_count(cfg, mode="train") > 1 and get_gpu_count(cfg, mode="test") == 1:
        weights = rename_ordered_dict_from_parallel(weights)
    if get_gpu_count(cfg, mode="train") == 1 and get_gpu_count(cfg, mode="test") > 1:
        weights = rename_ordered_dict_to_parallel(weights)
    model.load_state_dict(weights)

    dataloader = get_dataloader(cfg, samples_list)

    if not os.path.isdir(destination):
        os.makedirs(destination)

    infer(
        model,
        dataloader,
        output_types,
        destination,
    )


if __name__ == "__main__":
    args = parser()
    run_infer(
        args.cfg,
        args.checkpoint,
        args.samples_list,
        args.destination,
        args.outputs,
    )
