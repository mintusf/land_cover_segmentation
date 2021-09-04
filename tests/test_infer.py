import os

from infer import run_infer
from utils.utilities import split_sample_name
from utils.io_utils import get_lines_from_txt


def test_infer(module_dict, test_checkpoint, test_config):

    run_infer(
        module_dict["cfg_path"],
        test_checkpoint["path"],
        test_config.DATASET.LIST_TEST,
        module_dict["infer_directory"],
        ["alphablend"],
    )

    for sample_name in get_lines_from_txt(test_config.DATASET.LIST_TEST):
        roi_folder, area, _ = split_sample_name(sample_name)
        alphablend_folder = os.path.join(
            module_dict["infer_directory"], roi_folder, area
        )
        assert os.path.isfile(
            os.path.join(alphablend_folder, sample_name + "_alphablend.png")
        )
