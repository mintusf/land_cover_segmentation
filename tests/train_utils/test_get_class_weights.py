from torch import Tensor

from train_utils.losses import get_class_weights
from utils.io_utils import load_yaml


def test_get_class_weights(test_config):

    mask_config = load_yaml(test_config.DATASET.MASK.CONFIG)
    weights = get_class_weights(test_config, mask_config)

    assert isinstance(weights, Tensor)
    # np.testing.assert_almost_equal(weights, weights_target, decimal=6)
