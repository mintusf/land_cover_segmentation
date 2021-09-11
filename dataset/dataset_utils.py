import json

import numpy as np
import pandas as pd
from yacs.config import CfgNode

from utils.raster_utils import raster_to_np
from utils.io_utils import get_lines_from_txt, load_yaml
from utils.utilities import get_raster_filepath


def build_mask_single_channel(
    whole_mask: np.array, channel_int: int, labels_config: dict
) -> np.array:
    """Builds single mask from a single channel given labels parsing config.

    Args:
        whole_mask (np.array): 3D array of mask.
        channel_int (int): Channel index.
        labels_config (dict): Parsing config for labels.
                              It includes class_int: list of labels pairs.

    Returns:
        np.array: Generated mask, where all labels are set to corresponding class_int.
                  If label is not in labels_config, it is set to 0.
    """
    single_mask = whole_mask[channel_int, :, :]
    mask_out = np.zeros(single_mask.shape, dtype=np.uint8)
    for class_int, orig_labels in labels_config.items():
        masked_arr = np.any(
            np.stack([single_mask == orig_label for orig_label in orig_labels]),
            axis=0,
        )
        mask_out = np.where(
            masked_arr,
            class_int,
            mask_out,
        )

    return mask_out


def build_mask(whole_mask: np.array, mask_config: dict) -> np.array:
    """Builds mask from a mask parsing config.

    Args:
        whole_mask (np.array): 3D array of mask.
        mask_config (dict): Parsing config for mask.

    Returns:
        np.array: Generated mask with pixel-wise target classes.
    """
    maskname2int = mask_config["maskname2int"]
    channels_priority = mask_config["channels_priority"]
    selected_masks = mask_config["selected_masks"]

    mask_out = np.zeros((whole_mask.shape[1], whole_mask.shape[2]), dtype=np.uint8)

    for channel in channels_priority:
        channel_int = maskname2int[channel]
        channel_selected_masks = selected_masks[channel]
        single_mask = build_mask_single_channel(
            whole_mask, channel_int, channel_selected_masks
        )
        mask_out = np.where(mask_out == 0, single_mask, mask_out)

    return mask_out


def get_channels_in_count(cfg: CfgNode) -> int:
    """Returns the number of input channels given config"""
    return len(cfg.DATASET.INPUT.USED_CHANNELS)


def get_channels_out_count(cfg: CfgNode) -> int:
    """Returns the number of output channels given config"""
    labels_config = load_yaml(cfg.DATASET.MASK.CONFIG)
    return len(labels_config["class2label"])


def get_counts_from_raster(target_raster_path, mask_config, classes_count):
    target_np = raster_to_np(target_raster_path)
    transformed_mask = build_mask(target_np, mask_config)
    mask_unique = np.unique(transformed_mask, return_counts=True)
    mask_counts = np.zeros(classes_count)
    for i, count in zip(mask_unique[0], mask_unique[1]):
        mask_counts[i] = count

    return mask_counts


def get_class_distribution(cfg, mask_config, subgrids_list):

    class2label = mask_config["class2label"]
    classes_count = len(class2label)
    dataset_root = cfg.DATASET.ROOT
    target_sensor_name = cfg.DATASET.MASK.SENSOR

    classes_counts_single_mode = np.zeros(classes_count)
    # Get target tensor
    for sample_name in subgrids_list:
        target_raster_path = get_raster_filepath(
            dataset_root, sample_name, target_sensor_name
        )

        mask_counts = get_counts_from_raster(
            target_raster_path, mask_config, classes_count
        )
        classes_counts_single_mode += mask_counts

    classes_counts_single_mode = {
        i: classes_counts_single_mode[i] for i in range(classes_count)
    }

    return classes_counts_single_mode


def build_classes_distribution_json(cfg, mask_config):

    classes_counts_all = {}

    for mode in ["train", "val", "test"]:
        subgrids_list = eval(f"get_lines_from_txt(cfg.DATASET.LIST_{mode.upper()})")

        classes_counts_single_mode = get_class_distribution(
            cfg, mask_config, subgrids_list
        )

        classes_counts_all[mode] = classes_counts_single_mode

    json_save_path = cfg.DATASET.CLASSES_COUNT_JSON
    with open(json_save_path, "w") as f:
        json.dump(classes_counts_all, f)


def get_classes_counts_from_json(cfg, mode):

    json_save_path = cfg.DATASET.CLASSES_COUNT_JSON
    with open(json_save_path, "r") as f:
        classes_counts_all = json.load(f)

    return classes_counts_all[mode]


def build_masks_metadata_df(cfg, mask_config):

    subgrids_list_train = get_lines_from_txt(cfg.DATASET.LIST_TRAIN)
    subgrids_list_val = get_lines_from_txt(cfg.DATASET.LIST_VAL)
    subgrids_list_test = get_lines_from_txt(cfg.DATASET.LIST_TEST)
    subgrids_list_all = subgrids_list_train + subgrids_list_val + subgrids_list_test

    dataset_root = cfg.DATASET.ROOT
    target_sensor_name = cfg.DATASET.MASK.SENSOR
    class2label = mask_config["class2label"]
    classes_count = len(class2label)

    all_counts = []
    # Get target tensor
    for sample_name in subgrids_list_all:
        target_raster_path = get_raster_filepath(
            dataset_root, sample_name, target_sensor_name
        )

        mask_counts = get_counts_from_raster(
            target_raster_path, mask_config, classes_count
        )

        all_counts.append(mask_counts)

    all_counts = np.stack(all_counts)
    df_labels_counts = pd.DataFrame(
        all_counts,
        index=subgrids_list_all,
        columns=[class2label[i] for i in range(classes_count)],
    )

    df_save_path = mask_config["MASKS_METADATA_PATH"]
    df_labels_counts.to_csv(df_save_path, index_label="subgrid_name")
