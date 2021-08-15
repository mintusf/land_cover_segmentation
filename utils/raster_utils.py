from typing import List, Tuple, Union

import cv2
import numpy as np
import rasterio as rio
from rasterio.io import DatasetReader
import torch


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


def raster_to_np(
    raster_path: str, bands: Tuple[int] = None, dtype=np.float32
) -> np.array:
    """Convert img raster to numpy array. Raster can have any number of bands.
    Args:
        raster_path (str): Path to WV .img file
        bands (Tuple[int]): Tuple of bands to extract
    Returns:
        np.array: raster converted into np.array
    """
    with rio.open(raster_path) as src:
        if bands is None:
            bands = [src.read(band_idx + 1) for band_idx in range(src.count)]
        else:
            bands = [src.read(band_idx + 1) for band_idx in bands]

    img_np = np.array(bands, dtype=dtype)

    return img_np


def convert_for_vis(
    raster_path: str,
    global_stats_dict: dict,
    all_channels: Tuple[str],
    bands_rgb: Tuple[int] = [3, 2, 1],
    target_size: Tuple[int] = [256, 256],
) -> np.array:
    img = raster_to_np(raster_path, bands_rgb)
    img = transpose_to_channels_first(img)
    img = cv2.resize(img, target_size)

    means = [global_stats_dict["means"][all_channels[channel]] for channel in bands_rgb]
    stds = [global_stats_dict["stds"][all_channels[channel]] for channel in bands_rgb]

    for channel in range(img.shape[2]):
        img[:, :, channel] = (img[:, :, channel] - means[channel]) / stds[channel]

    img = (img + 2) * 255 / 4

    img = img.astype(np.uint8)
    return img


def transpose_to_channels_first(np_arrray: np.array) -> np.array:
    """Expand np.array to 3-dimensions."""
    if np_arrray.ndim == 3:
        img_np = np.transpose(np_arrray, [1, 2, 0])
    return img_np


def np_to_torch(img_np: np.array, dtype=torch.float) -> torch.Tensor:
    """Convert np.array to torch.Tensor."""
    if img_np.dtype != np.float32:
        img_np = img_np.astype(np.float32)
    img_tensor = torch.from_numpy(img_np)

    img_tensor = img_tensor.type(dtype)

    return img_tensor


def raster_to_tensor(
    raster_path: str,
    bands: Union[Tuple[int], None] = None,
) -> torch.Tensor:
    """Convert img raster to torch.Tensor. Raster can have any number of bands.
    Args:
        raster_path (str): Path to the raster
        bands (Union[Tuple[int], None]): If given, includes bands to be extract.
                                         If None, all bands are extracted.
    Returns:
        torch.Tensor: Raster converted into tensor of shape (n_bands, height, width)
    """
    img_np = raster_to_np(raster_path, bands, dtype=np.float32)
    img_tensor = np_to_torch(img_np)

    return img_tensor


def get_stats(file: str) -> Tuple[np.array]:
    """Gets raster's mean and std for each channel.

    Args:
        file (str): Path to the raster

    Returns:
        Tuple[np.array]: (mean, std)
    """
    img_np = raster_to_np(file)
    image_stds = np.std(img_np, axis=(1, 2))
    image_means = np.mean(img_np, axis=(1, 2))
    return image_means, image_stds
