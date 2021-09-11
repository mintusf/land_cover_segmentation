import os

import pandas as pd

from dataset.dataset_utils import build_classes_distribution_json

from utils.io_utils import load_yaml, load_json, get_lines_from_txt


def test_build_masks_metadata_df(test_config):
    mask_config = load_yaml(test_config.DATASET.MASK.CONFIG)
    build_classes_distribution_json(test_config, mask_config)

    classes_distribution_json_path = test_config.DATASET.CLASSES_COUNT_JSON
    assert os.path.isfile(classes_distribution_json_path)

    classes_distribution = load_json(classes_distribution_json_path)

    assert list(classes_distribution.keys()) == ["train", "val", "test"]

    class2label = mask_config["class2label"]
    for mode in ["train", "val", "test"]:
        assert list(classes_distribution[mode].keys()) == [
            str(i) for i in range(len(class2label))
        ]
