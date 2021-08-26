from typing import Dict, Union

from comet_ml import Experiment
import numpy as np

from config.default import get_cfg_from_file, CfgNode


def init_comet_logging(cfg_path: str) -> Experiment:
    """Initialize comet logging."""

    cfg = get_cfg_from_file(cfg_path)
    if not cfg.TRAIN.USE_COMET:
        return None

    experiment = Experiment(
        auto_output_logging="simple",
        auto_param_logging=False,
        auto_metric_logging=False,
        project_name="sentinel_land_cover",
    )
    experiment.set_name(cfg_path)
    experiment.log_parameter("Config name", cfg_path)
    experiment.add_tags(cfg.TRAIN.COMET_TAGS)

    return experiment
