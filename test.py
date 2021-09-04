import argparse
import os
from utils.utilities import get_gpu_count

import cv2
import torch

from config.default import get_cfg_from_file
from dataset import get_dataloader
from models import get_model
from train_utils import load_checkpoint, get_loss, model_validation
from utils.io_utils import load_yaml, load_json
from utils.raster_utils import convert_np_for_vis
from utils.visualization_utils import create_alphablend


def parser():
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="Train the model")

    parser.add_argument(
        "--cfg",
        help="Path to the config file defining testing",
        type=str,
        default="/workspace/config/firstrun_focal.yml",
    )

    parser.add_argument(
        "--checkpoint",
        help="Path to the config file",
        type=str,
        default="/workspace/weights/from_aws/cfg_firstrun_focal_bestloss.pth",
    )

    parser.add_argument(
        "--destination",
        help="Path to the folder or text file containing list of images",
        type=str,
        default="/data/masks",
    )

    parser.add_argument(
        "--add_alphablend",
        action="store_false",
        help="Whether alphablend should be generated",
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


def run_testings(
    cfg_path: str, checkpoint: str, destination: str, add_alphablend: bool
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
    criterion = get_loss(cfg)

    dataloader = get_dataloader(cfg, "test")
    mask_config = load_yaml(cfg.DATASET.MASK.CONFIG)

    if not os.path.isdir(destination):
        os.makedirs(destination)

    loss, inputs, preds, names = model_validation(
        model, criterion, dataloader, return_tensors=True
    )

    preds = torch.argmax(preds, dim=1)
    print(f"Loss on test set: {loss['val_loss']}")
    print(f"Recall on test set: {loss['recall']}")
    print(f"Precision on test set: {loss['precision']}")
    print(f"F1 score on test set: {loss['f1']}")

    if add_alphablend:
        for idx in range(inputs.shape[0]):
            input_img = inputs[idx].cpu().numpy()
            input_img = input_img[(1, 2, 3), :, :]
            input_img = convert_np_for_vis(input_img)
            mask = preds[idx].cpu().numpy()
            name = names[idx]
            alphablend_path = os.path.join(destination, f"{name}_alphablend.png")
            alphablended = create_alphablend(input_img, mask, mask_config)
            cv2.imwrite(alphablend_path, alphablended)


if __name__ == "__main__":
    args = parser()
    run_testings(args.cfg, args.checkpoint, args.destination, args.add_alphablend)