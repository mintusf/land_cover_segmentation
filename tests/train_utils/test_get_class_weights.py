import numpy as np
import pandas as pd
from torch import Tensor

from train_utils.losses import get_class_weights


def test_get_class_weights():

    samples_list = ["sample0", "sample1", "sample2", "sample3", "sample4"]

    class2label = {0: "class0", 1: "class1", 2: "class2"}

    target_metadata = pd.DataFrame(
        {
            "sample": samples_list,
            "class0": [2, 4, 1, 1, 0],
            "class1": [1, 1, 1, 1, 1],
            "class2": [1, 3, 0, 5, 6],
        }
    )

    weights_target = [1 / 8, 1 / 5, 1 / 15]

    weights = get_class_weights(samples_list, class2label, target_metadata)

    assert isinstance(weights, Tensor)
    np.testing.assert_almost_equal(weights, weights_target, decimal=6)
