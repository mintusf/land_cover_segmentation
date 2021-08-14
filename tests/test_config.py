
def test_config(test_config):

    assert "MODEL" in test_config and "DATASET" in test_config
    assert test_config.MODEL.TYPE == "DeepLab"
    assert test_config.DATASET.ROOT == "./tests/dataset/dummy_dataset"
