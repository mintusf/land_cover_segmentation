from torch import Tensor

from train_utils.losses import get_class_weights


def test_get_class_weights(test_config):

    weights = get_class_weights(test_config)

    assert isinstance(weights, Tensor)
    # np.testing.assert_almost_equal(weights, weights_target, decimal=6)
