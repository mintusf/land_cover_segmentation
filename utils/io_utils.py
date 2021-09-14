import json
import random
from typing import List
import yaml
from yaml import CLoader as Loader


def get_lines_from_txt(
    txt_path: str,
    shuffle: bool = False,
    removed_subgrids=["ROIs1158_spring_24", "ROIs1158_spring_26"],
) -> List[str]:
    """Generates list of filenames from a txt file"""
    try:
        with open(txt_path, "r") as f:
            all_filenames = f.readlines()
        all_filenames = [filename.strip("\n") for filename in all_filenames]
        if shuffle:
            random.shuffle(all_filenames)
        for subgrid in removed_subgrids:
            all_filenames = [
                filename for filename in all_filenames if subgrid not in filename
            ]
    except FileNotFoundError:
        all_filenames = []
    return all_filenames


def load_json(json_path: str) -> dict:
    """Loads json file"""
    with open(json_path, "r") as f:
        json_data = json.load(f)
    return json_data


def load_yaml(yaml_path: str) -> dict:
    """Loads yaml file"""
    with open(yaml_path, "r") as f:
        yaml_data = yaml.load(f, Loader=Loader)
    return yaml_data
