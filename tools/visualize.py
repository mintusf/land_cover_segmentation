import argparse
import os
import random
from typing import List
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from config.default import get_cfg_from_file
from utils.io_utils import get_lines_from_txt
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
        default="config/weighted_loss.yml",
        type=str,
        help="root directory in which files area searched for",
    )

    parser.add_argument(
        "--count",
        default=100,
        type=int,
        help="How many samples to visualized (selected randomly)",
    )

    parser.add_argument(
        "--destination",
        default="/data/seg_data/visualization",
        type=str,
        help="path to save results",
    )

    parser.add_argument("--labels", nargs="+", default=["wetlands"], type=str, help="")
    parser.add_argument("--threshold", default=3000, type=int, help="")
    parser.add_argument(
        "--all_labels",
        action="store_false",
        help="Whether samples with all selected labels should be visualized",
    )
    parser.add_argument(
        "--metadata_path",
        default="/data/seg_data/training_labels.csv",
        type=str,
        help="path to metadata with labels count",
    )

    args = parser.parse_args()

    return args


def get_samples_by_label(
    metadata_path: str, labels: List[str], threshold: int, all_labels
) -> List[str]:
    """Get samples for which any label from labels have more pixels than threshold

    Args:
        metadata_path: path to csv file with metadata
        labels: list of labels
        threshold: threshold for pixel count

    Returns:
        List[str]: list of samples
    """
    metadata = pd.read_csv(metadata_path)
    labels_filters = []
    for label in labels:
        label_filter = metadata[label] > threshold
        labels_filters.append(label_filter)

    if all_labels:
        labels_filter = pd.concat(labels_filters, axis=1).all(axis=1)
    else:
        labels_filter = pd.concat(labels_filters, axis=1).any(axis=1)
    samples = metadata[labels_filter]["sample"].values
    return samples


if __name__ == "__main__":
    args = parse_args()
    cfg_path = args.config
    destination = args.destination
    os.makedirs(destination, exist_ok=True)
    cfg = get_cfg_from_file(cfg_path)
    samples = get_lines_from_txt(cfg.DATASET.LIST_TRAIN)
    samples_by_label = get_samples_by_label(
        args.metadata_path, args.labels, args.threshold, args.all_labels
    )
    samples = list(set(samples) & set(samples_by_label))

    random.seed(42)
    random.shuffle(samples)
    samples = samples[: args.count]

    for sample in samples:
        savepath = f"{destination}/{sample}.png"
        vis_sample(sample, cfg, savepath)
