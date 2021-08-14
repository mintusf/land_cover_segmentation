import numpy as np

from config.default import CfgNode
from utils.io_utils import load_yaml


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
