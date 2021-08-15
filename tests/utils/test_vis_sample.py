import os

from utils.io_utils import get_lines_from_txt
from utils.visualization_utils import vis_sample
from utils.utilities import build_dataset_stats_json_from_cfg


def test_vis_sample(test_config):
    build_dataset_stats_json_from_cfg(test_config)
    sample = get_lines_from_txt(test_config.DATASET.LIST_TRAIN)[0]
    savepath = "tests/utils/test_vis.png"
    vis_sample(sample, test_config, savepath)

    assert os.path.isfile(savepath)
