import os
from yacs.config import CfgNode

_C = CfgNode()

_C.IS_TEST = False

_C.MODEL = CfgNode()
_C.MODEL.TYPE = "DeepLab"

_C.DATASET = CfgNode()
_C.DATASET.ROOT = os.path.join("/data")
_C.DATASET.LIST_TRAIN = os.path.join("config", "dataset", "lists", "train.txt")
_C.DATASET.LIST_VAL = os.path.join("config", "dataset", "lists", "val.txt")
_C.DATASET.LIST_TEST = os.path.join("config", "dataset", "lists", "test.txt")
_C.DATASET.INPUT = CfgNode()
_C.DATASET.INPUT.SENSOR = "s2"

_C.DATASET.INPUT.CHANNELS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8a",
    "B9",
    "B10",
    "B11",
    "B12",
]
_C.DATASET.INPUT.USED_CHANNELS = [3, 2, 1]
_C.DATASET.INPUT.STATS_FILE = os.path.join("config", "dataset", "stats", "default.json")
_C.DATASET.MASK = CfgNode()
_C.DATASET.MASK.SENSOR = "lc"
_C.DATASET.MASK.CONFIG = os.path.join(
    "config", "dataset", "mask_configs", "default.yml"
)

_C.TRAIN = CfgNode()
_C.TRAIN.EPOCHS = 100
_C.TRAIN.RESUME_CHECKPOINT = ""
_C.TRAIN.BATCH_SIZE_PER_DEVICE = 8

# Can be
# * `cpu`
# * `cuda:N` (one device)
# * `cuda:N1,N2` (multiple devices)
# * `cuda:all` (all available devices)
_C.TRAIN.DEVICE = "cuda:0"

_C.TRAIN.WORKERS = 8
_C.TRAIN.SHUFFLE = True
_C.TRAIN.SEED = 42
_C.TRAIN.LOSS = "categorical_crossentropy"
_C.TRAIN.OPTIMIZER = "adam"
_C.TRAIN.LR = 0.001
_C.TRAIN.WEIGHT_DECAY = 0.0005
_C.TRAIN.VERBOSE_STEP = 10
_C.TRAIN.VAL_PER_EPOCH = 6
_C.TRAIN.SCHEDULER = CfgNode()
_C.TRAIN.SCHEDULER.TYPE = "ReduceLROnPlateau"
_C.TRAIN.SCHEDULER.FACTOR = 0.1
_C.TRAIN.SCHEDULER.PATIENCE = 4
_C.TRAIN.WEIGHTS_FOLDER = "tests/weights"
_C.TRAIN.USE_COMET = True
_C.TRAIN.COMET_TAGS = ["experiment", "cross_entropy", "focal_loss"]


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


def get_cfg_from_file(filepath: str) -> CfgNode:
    """Load a config file and return a CfgNode object"""
    cfg = get_cfg_defaults()
    cfg.merge_from_file(filepath)
    cfg.freeze()

    return cfg
