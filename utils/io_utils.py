import json
from typing import List
import random
import yaml
from yaml import CLoader as Loader


def get_lines_from_txt(txt_path: str, shuffle: bool = False) -> List[str]:
    """Generates list of filenames from a txt file"""
    try:
        with open(txt_path, "r") as f:
            all_filenames = f.readlines()
        all_filenames = [filename.strip("\n") for filename in all_filenames]
        if shuffle:
            random.shuffle(all_filenames)
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
