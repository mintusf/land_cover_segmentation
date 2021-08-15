import argparse
import os
import random
import sys

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
        default="config/firstrun.yml",
        type=str,
        help="root directory in which files area searched for",
    )

    parser.add_argument(
        "--samples",
        default=100,
        type=int,
        help="How many samples to visualized (selected randomly)",
    )

    parser.add_argument(
        "--destination",
        default="/data/visualization",
        type=str,
        help="path to save results",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    cfg_path = args.config
    destination = args.destination
    os.makedirs(destination, exist_ok=True)
    cfg = get_cfg_from_file(cfg_path)
    samples = get_lines_from_txt(cfg.DATASET.LIST_TRAIN)

    random.seed(42)
    selected_samples = random.choices(samples, k=args.samples)

    for sample in selected_samples:
        savepath = f"{destination}/{sample}.png"
        vis_sample(sample, cfg, savepath)
