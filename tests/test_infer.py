import os

from infer import run_infer
from utils.utilities import split_sample_name
from utils.io_utils import get_lines_from_txt


def test_infer(module_dict, test_checkpoint, test_config):

    infer_modes = ["alphablend", "raster", "alphablended_raster"]
    run_infer(
        module_dict["cfg_path"],
        test_checkpoint["path"],
        test_config.DATASET.LIST_TEST,
        module_dict["infer_directory"],
        infer_modes,
    )

    for sample_name in get_lines_from_txt(test_config.DATASET.LIST_TEST):
        roi_folder, area, _ = split_sample_name(sample_name)
        for infer_mode in infer_modes:
            folder = os.path.join(
                module_dict["infer_directory"], infer_mode, roi_folder, area
            )
            ext = "tif" if "raster" in infer_mode else "png"
            assert os.path.isfile(
                os.path.join(folder, sample_name + f"_{infer_mode}.{ext}")
            )
