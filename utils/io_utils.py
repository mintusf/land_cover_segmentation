from typing import List


def get_lines_from_txt(txt_path: str) -> List[str]:
    """Generates list of filenames from a txt file"""
    try:
        with open(txt_path, "r") as f:
            all_filenames = f.readlines()
        all_filenames = [filename.strip("\n") for filename in all_filenames]
    except FileNotFoundError:
        all_filenames = []
    return all_filenames
