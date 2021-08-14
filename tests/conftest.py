import os
import pytest

from config.default import get_cfg_from_file

test_config_path = os.path.join("config", "tests.yml")


@pytest.fixture(scope="session")
def test_config():
    cfg = get_cfg_from_file(test_config_path)
    return cfg


def with_class_json(func):
    def wrapper():

        cfg = get_cfg_from_file(test_config_path)
        assert not os.path.isfile(cfg.DATASET.INPUT.STATS_FILE)

        func(cfg)

        # Test if stats json exists
        assert os.path.isfile(cfg.DATASET.INPUT.STATS_FILE)
        os.remove(cfg.DATASET.INPUT.STATS_FILE)

    return wrapper
