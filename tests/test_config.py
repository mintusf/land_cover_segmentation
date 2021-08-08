import os

from config.default import get_cfg_defaults


def test_config():
    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join("config", "tests.yml"))
    cfg.freeze()
    print(type(cfg))

    assert "MODEL" in cfg and "DATASET" in cfg
    assert cfg.MODEL.TYPE == "DeepLab"
    assert cfg.DATASET.ROOT == "/data"