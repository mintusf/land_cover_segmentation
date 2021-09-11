import os

import pandas as pd

from dataset.dataset_utils import build_masks_metadata_df

from utils.io_utils import load_yaml, get_lines_from_txt


def test_build_masks_metadata_df(test_config):
    mask_config = load_yaml(test_config.DATASET.MASK.CONFIG)
    build_masks_metadata_df(test_config, mask_config)

    assert os.path.isfile(mask_config["MASKS_METADATA_PATH"])
    df = pd.read_csv(mask_config["MASKS_METADATA_PATH"], index_col=0)

    subgrids_list_train = get_lines_from_txt(test_config.DATASET.LIST_TRAIN)
    subgrids_list_val = get_lines_from_txt(test_config.DATASET.LIST_VAL)
    subgrids_list_test = get_lines_from_txt(test_config.DATASET.LIST_TEST)
    subgrids_list_all = subgrids_list_train + subgrids_list_val + subgrids_list_test

    assert list(df.index) == subgrids_list_all

    class2label = mask_config["class2label"]
    assert list(df.columns) == [class2label[i] for i in range(len(class2label))]
