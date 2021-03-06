import argparse
from utils.utilities import get_gpu_count

import pandas as pd
import torch

from config.default import get_cfg_from_file
from dataset import get_dataloader
from models import get_model
from models.models_utils import (
    rename_ordered_dict_from_parallel,
    rename_ordered_dict_to_parallel,
)
from train_utils import load_checkpoint, get_loss, model_validation
from utils.io_utils import load_yaml


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

    return parser.parse_args()


def run_testings(cfg_path: str, checkpoint: str):

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

    metrics = model_validation(model, criterion, dataloader, return_ave=False)

    print(f"Loss on test set: {metrics['val_loss']}")

    labels = [mask_config["class2label"][i] for i in sorted(mask_config["class2label"])]
    metrics_df = pd.DataFrame(
        {
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "f1": metrics["f1"],
        },
        index=labels,
    )
    print(metrics_df)


if __name__ == "__main__":
    args = parser()
    run_testings(args.cfg, args.checkpoint)
