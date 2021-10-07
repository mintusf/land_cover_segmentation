import argparse
from typing import List
import os

import torch
from torch.utils.data import DataLoader
from torch.nn import Module

from config.default import get_cfg_from_file
from dataset import get_dataloader
from models import get_model
from models.models_utils import (
    rename_ordered_dict_from_parallel,
    rename_ordered_dict_to_parallel,
)
from train_utils import load_checkpoint
from utils.utilities import get_gpu_count
from utils.infer_utils import generate_outputs, prepare_raster_for_inference
from utils.io_utils import get_lines_from_txt


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
        default="test",
    )

    parser.add_argument(
        "--destination",
        help="Path for saving results",
        type=str,
        default="/data/seg_data/inference",
    )

    parser.add_argument(
        "--outputs",
        nargs="+",
        default=["raster"],
        help="What kind of outputs to generate "
        + "from ['alphablend','raster','alphablended_raster', 'raw_raster']",
    )

    return parser.parse_args()


def infer(
    model: Module,
    dataloader: DataLoader,
    output_types: List[str],
    destination: str,
):
    """Evaluates test dataset and saves predictions if needed

    Args:
        model (Module): Model to use for inference
        dataloader (DataLoader): Dataloader for inference
        output_types (List[str]): List of output types.
                                  Supported types:
                                    * alphablend (img and predicted mask)
        destination (str): Path to save results

    Returns:
        dict: Generates and saves predictions in desired format
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

                generate_outputs(
                    output_types,
                    destination,
                    input_img,
                    mask,
                    name,
                    mask_config,
                    dataloader,
                )


def run_infer(
    cfg_path: str,
    checkpoint: str,
    samples_list_path: str,
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

    if samples_list_path not in ["train", "val", "test"]:
        samples_list = get_lines_from_txt(samples_list_path)
        samples_to_infer = []
        for sample_path in samples_list:
            cropped_samples_paths = prepare_raster_for_inference(
                sample_path, crop_size=[256, 256]
            )
            samples_to_infer.extend(cropped_samples_paths)

        with open(cfg.TEST.INFER_SAMPLES_LIST_PATH, "w") as f:
            for file in samples_to_infer:
                f.write(file + "\n")

        samples_list_path = cfg.TEST.INFER_SAMPLES_LIST_PATH

    dataloader = get_dataloader(cfg, samples_list_path)

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
