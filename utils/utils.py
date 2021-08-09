import os


def split_sample_name(sample_name: str) -> str:
    """Split sample name into ROI folder name, area, and subgrid ID."""
    parts = sample_name.split("_")
    roi_folder_name = "_".join(parts[:2])
    area = parts[2]
    subgrid_id = parts[3]
    return roi_folder_name, area, subgrid_id


def get_area_foldername(sensor: str, area: str) -> str:
    """Get area foldername given sensor and area"""
    return f"{sensor}_{area}"


def get_raster_filepath(rootdir: str, sample_name: str, sensor: str) -> str:
    """Get raster filepath given rootdir, sample name, and sensor
    Args:
        rootdir (str): root directory of the dataset
        sample_name (str): sample name, e.g "ROIs2017_winter_27_p36"
        sensor (str): sensor name

    Returns:
        str: raster filepath
    """
    roi_folder_name, area, subgrid_id = split_sample_name(sample_name)
    folder = os.path.join(rootdir, roi_folder_name, get_area_foldername(sensor, area))
    filename = f"{roi_folder_name}_{sensor}_{area}_{subgrid_id}.tif"
    return os.path.join(folder, filename)
