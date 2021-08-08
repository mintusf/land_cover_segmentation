from typing import List


def get_coord_from_raster(raster) -> List[float]:
    """[summary]

    Args:
        raster_filename (str): [description]

    Returns:
        List[float]: [description]
    """
    (x0, y0) = raster.xy(0, 0)
    (x1, y1) = raster.xy(raster.height, raster.width)

    return [x0, y0, x1, y1]
