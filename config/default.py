import os
from yacs.config import CfgNode

_C = CfgNode()

_C.MODEL = CfgNode()
_C.MODEL.TYPE = "DeepLab"

_C.DATASET = CfgNode()
_C.DATASET.ROOT = os.path.join("/data")
_C.DATASET.LIST = os.path.join("config", "dataset", "lists", "test.txt")
_C.DATASET.MASK = CfgNode()
_C.DATASET.MASK.CONFIG = os.path.join(
    "config", "dataset", "mask_configs", "default.yml"
)


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
