from typing import List

from rasterio.io import DatasetReader


def get_coord_from_raster(raster: DatasetReader) -> List[float]:
    """Gets coordintes from raster

    Args:
        raster (DatasetReader): Rasterio dataset reader

    Returns:
        List[float]: Coordinates x0, y0, x1, y1
    """
    print(type(raster))
    (x0, y0) = raster.xy(0, 0)
    (x1, y1) = raster.xy(raster.height, raster.width)

    return [x0, y0, x1, y1]
