import numpy as np

from dataset.dataset_utils import build_mask
from utils.io_utils import load_yaml


def test_build_mask(test_config):
    """
    Given config:
        mask_1:
            1: [1]
            2: [2, 3, 4]
        mask_2:
            3: [1]
            4: [2, 4]
            0: [0]
    And masks priority ["mask_1", "mask_2"].

    Following mask is transformed:
        # channel 0 ("mask_1"):
            1 1 1 0 0
            0 5 5 0 0
            0 5 5 0 0
            0 0 2 2 4
            0 0 3 2 4

        # channel 1 ("mask_2"):
            1 1 0 0 0
            1 1 3 2 4
            1 1 3 2 4
            1 1 3 2 4
            1 1 3 2 4


    Target mask_out, where all labels are set to corresponding class_int, is as follows:
        1 1 1 0 0
        3 3 0 4 4
        3 3 0 4 4
        3 3 2 2 2
        3 3 2 2 2
    """

    mask_config = load_yaml(test_config.DATASET.MASK.CONFIG)

    whole_mask_dummy_channel_1 = np.array(
        [
            [1, 1, 1, 0, 0],
            [0, 5, 5, 0, 0],
            [0, 5, 5, 0, 0],
            [0, 0, 2, 2, 4],
            [0, 0, 3, 2, 4],
        ]
    )

    whole_mask_dummy_channel_2 = np.array(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 3, 2, 4],
            [1, 1, 3, 2, 4],
            [1, 1, 3, 2, 4],
            [1, 1, 3, 2, 4],
        ]
    )

    whole_mask = np.stack(
        [whole_mask_dummy_channel_1, whole_mask_dummy_channel_2], axis=0
    )

    mask_out_target = np.array(
        [
            [1, 1, 1, 0, 0],
            [3, 3, 0, 4, 4],
            [3, 3, 0, 4, 4],
            [3, 3, 2, 2, 2],
            [3, 3, 2, 2, 2],
        ]
    )

    mask_out = build_mask(whole_mask, mask_config)

    assert np.array_equal(mask_out, mask_out_target)
