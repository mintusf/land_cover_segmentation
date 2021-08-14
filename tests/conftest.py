import os
import pytest

from config.default import get_cfg_from_file


@pytest.fixture(scope="session")
def test_config():
    cfg = get_cfg_from_file(os.path.join("config", "tests.yml"))
    return cfg
