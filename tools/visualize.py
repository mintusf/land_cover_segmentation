import argparse
import os
import random
from typing import List
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from config.default import get_cfg_from_file
from dataset.dataset_utils import build_masks_metadata_df
from utils.io_utils import get_lines_from_txt, load_yaml
from utils.visualization_utils import vis_sample


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Calculate global mean and std channel-wise for subgrids"
    )

    parser.add_argument(
        "--config",
        default="config/tests.yml",
        type=str,
        help="root directory in which files area searched for",
    )

    parser.add_argument(
        "--count",
        default=50,
        type=int,
        help="How many samples to visualized per label (selected randomly)",
    )

    parser.add_argument(
        "--destination",
        default="/data/seg_data/visualization",
        type=str,
        help="path to save results",
    )

    parser.add_argument("--labels", nargs="+", default=["class_1"], type=str, help="")
    parser.add_argument("--threshold", default=6000, type=int, help="")

    args = parser.parse_args()

    return args


def get_samples_by_label(metadata_path: str, label: str, threshold: int) -> List[str]:
    """Get samples for which any label from labels have more pixels than threshold

    Args:
        metadata_path: path to csv file with metadata
        labels: list of labels
        threshold: threshold for pixel count

    Returns:
        List[str]: list of samples
    """
    metadata = pd.read_csv(metadata_path)

    label_filter = metadata[label] > threshold

    samples = metadata[label_filter]["subgrid_name"].values
    return samples


if __name__ == "__main__":
    args = parse_args()
    cfg_path = args.config
    destination = args.destination
    cfg = get_cfg_from_file(cfg_path)
    samples_all = get_lines_from_txt(cfg.DATASET.LIST_TRAIN)

    mask_config = load_yaml(cfg.DATASET.MASK.CONFIG)
    metadata_path = mask_config["MASKS_METADATA_PATH"]
    if not os.path.isfile(metadata_path):
        build_masks_metadata_df(cfg, mask_config)

    for label in args.labels:
        print(label)
        samples_by_label = get_samples_by_label(metadata_path, label, args.threshold)
        print(len(samples_by_label))
        samples_by_label = list(set(samples_all) & set(samples_by_label))

        random.seed(42)
        random.shuffle(samples_by_label)
        samples_by_label = samples_by_label[: args.count]

        for sample in samples_by_label:
            os.makedirs(f"{destination}/{label}", exist_ok=True)
            savepath = f"{destination}/{label}/{sample}.png"
            vis_sample(sample, cfg, savepath)
