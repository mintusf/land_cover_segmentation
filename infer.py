import argparse
import glob
import os

from torch.nn import Module

from config.default import get_cfg_from_file
from dataset import get_dataloader
from models import get_model
from train_utils import load_checkpoint
from utils.io_utils import get_lines_from_txt


def parser():
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="Train the model")

    parser.add_argument(
        "--list",
        help="Path to the text file containing list of images",
        type=str,
        default="/workspace/config/dataset/lists/test.txt",
    )

    parser.add_argument(
        "--checkpoint",
        help="Path to the config file",
        type=str,
        default="/workspace/weights/cfg_firstrun_focal_epoch_20.pth",
    )

    parser.add_argument(
        "--target",
        help="Path to the folder or text file containing list of images",
        type=str,
        default="/data/masks",
    )

    parser.add_argument(
        "--add_alphablend",
        action="store_true",
        help="Whether alphablend should be generated",
    )

    return parser.parse_args()


def single_inference(model: Module, input_filepath: str):
    """Infer a single image."""
    pass


def create_alphablend(input, output, name, alphablend_path):
    pass


def save_tensor_as_raster(tensor, raster_path):
    pass


def infer(target: str, checkpoint: str, destination: str, add_alphablend: bool):

    # Build the model
    _, weights, _, _, cfg_path = load_checkpoint(checkpoint)
    cfg = get_cfg_from_file(cfg_path)
    model = get_model(cfg)
    model.load_state_dict(weights)

    dataloader = get_dataloader(cfg, "test")

    if not os.path.isdir(target):
        os.makedirs(target)

    for batch in dataloader:
        inputs, names = batch["input"], batch["name"]
        outputs = model(inputs)
        for input, output, name in zip(inputs, outputs, names):
            if add_alphablend:
                alphablend_path = ""
                create_alphablend(input, output, name, alphablend_path)
            output_path = ""
            save_tensor_as_raster(output, output_path)


if __name__ == "__main__":
    args = parser()
    infer(args.target, args.checkpoint, args.destination, args.add_alphablend)
