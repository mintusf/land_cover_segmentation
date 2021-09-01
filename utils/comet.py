from typing import Dict, Union

from comet_ml import Experiment
import numpy as np
from torch import Tensor

from config.default import get_cfg_from_file, CfgNode
from utils.utilities import get_class_labels_ordered, get_train_step


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


def log_metrics_comet(
    cfg: CfgNode,
    metrics: Dict[str, Union[float, Tensor]],
    experiment: Experiment,
    epoch: int,
    batch_no: int,
) -> None:
    """Log metrics to comet.

    Args:
        cfg (CfgNode): Config
        metrics (Dict[str, Union[float, Tensor]]): Dict of metrics to log
        experiment (Experiment): comet_ml experiment
        epoch (int): Current epoch
        batch_no (int): Current batch number
    """
    if experiment is not None:
        step = get_train_step(cfg, batch_no, epoch - 1)

        for metric_str, value in metrics.items():
            if "confusion_matrix" not in metric_str:
                experiment.log_metric(metric_str, value, step=step, epoch=epoch)
            else:
                labels = get_class_labels_ordered(cfg)
                experiment.log_confusion_matrix(
                    matrix=(value / 1000).astype(np.int32).tolist(),
                    title=metric_str + " (Count in 1K)",
                    max_example_per_cell=200000,
                    labels=labels,
                    file_name=f"{metric_str}-epoch{epoch}-batch{batch_no}.json",
                )
