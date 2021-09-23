from typing import List, Tuple, Union
import os

import cv2
import numpy as np
import rasterio as rio
from rasterio import mask
from rasterio.io import DatasetReader
from shapely.geometry import Polygon
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


def convert_np_for_vis(
    img: np.array,
    target_size: Tuple[int] = [256, 256],
) -> np.array:
    """Convert np.array to open-cv format.

    Args:
        img (np.array): np.array to be converted
        target_size (Tuple[int], optional): Size of returned image.
                                            Defaults to [256, 256].

    Returns:
        [np.array]: Converted np.array
    """
    img = transpose_to_channels_first(img)
    img = cv2.resize(img, target_size)

    img = np.clip(img, -2, 2)
    img = (img + 2) * 255 / 4
    img = img.astype(np.uint8)
    return img


def convert_raster_for_vis(
    raster_path: str,
    bands_rgb: Tuple[int] = [3, 2, 1],
) -> np.array:
    """Given path to raster and RGB bands, converts and returns saveable image.

    Args:
        raster_path (str): Path to the raster
        bands_rgb (Tuple[int], optional): Indication of RGB bands.
                                          Defaults to [3, 2, 1].

    Returns:
        np.array: Image converted to np.array saveable with open-cv
    """
    img = raster_to_np(raster_path, bands_rgb)

    img = convert_np_for_vis(img)

    return img


def transpose_to_channels_first(np_arrray: np.array) -> np.array:
    """Transpose np.array to open-cv format"""
    if np_arrray.ndim == 3:
        np_arrray = np.transpose(np_arrray, [1, 2, 0])
    return np_arrray


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


def get_stats(file: str, channels_count: int) -> Tuple[np.array]:
    """Gets raster's mean and std for each channel.

    Args:
        file (str): Path to the raster
        channels_count (int): Target number of channels in the raster
                              If actual number is smaller, stats set to np.nan

    Returns:
        Tuple[np.array]: (mean, std)
    """
    img_np = raster_to_np(file)
    image_stds = np.std(img_np, axis=(1, 2))
    image_means = np.mean(img_np, axis=(1, 2))

    # Pad to channels_count
    if image_means.shape[0]:
        pad = np.empty(channels_count - image_stds.shape[0])
        pad.fill(np.nan)
        image_stds = np.concatenate([image_stds, pad])
        image_means = np.concatenate([image_means, pad])
    return image_means, image_stds


def np_to_raster(img_np: np.array, ref_img: str, savepath: str):
    """Convert np.array to raster and save
    Args:
        img_np (np.array): Image to be saved
        ref_img (str): Referenced raster
        savepath (str): Output raster savepath (tif format is recommended)
    """
    with rio.open(ref_img) as src:
        transform = src.transform
        size = (src.height, src.width)

    with rio.open(
        savepath,
        "w",
        driver="GTiff",
        dtype=img_np.dtype,
        height=size[0],
        width=size[1],
        count=3,
        crs=src.crs,
        transform=transform,
    ) as dst:
        dst.write(img_np)


def is_cropped(input_raster: str, crop_size: List[int]):
    with rio.open(input_raster) as src:
        height, width = (src.height, src.width)
        if height < crop_size[0] or width < crop_size[1]:
            raise ValueError(
                "Raster cannot have smaller size than crop size. "
                + f"Raster's size is [{height}, {width}], crop size: {crop_size}"
            )
        if height > crop_size[0] or width > crop_size[1]:
            return True
        else:
            return False


def crop_raster(input_raster: str, dest_dir: str, crop_size: List[int]):
    """Crop raster into subgrids
    Args:
        input_img (str): Path to raster file
        dest_dir (str): Destination directory. Must not exist.
        crop_size (List[int]): dimensions of the subgrid
    """
    files = []

    with rio.open(input_raster) as src:
        height, width = (src.height, src.width)

        lat_crop_num = height // crop_size[0]
        long_crop_num = width // crop_size[1]

        for lat_idx in range(lat_crop_num):
            for long_idx in range(long_crop_num):

                x_min = lat_idx * crop_size[0]
                y_min = long_idx * crop_size[1]
                (west, north) = src.xy(x_min, y_min)

                x_max = (lat_idx + 1) * crop_size[0] - 1
                y_max = (long_idx + 1) * crop_size[1] - 1
                (east, south) = src.xy(x_max, y_max)

                polygon = Polygon(
                    [(west, north), (east, north), (east, south), (west, south)]
                )

                out_image, out_transform = mask.mask(
                    src, [polygon], crop=True, all_touched=True
                )
                out_meta = src.meta

                out_meta.update(
                    {
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                    }
                )

                raster_name = os.path.splitext(os.path.split(input_raster)[1])[0]
                out_path = os.path.join(
                    dest_dir, f"{raster_name}_{lat_idx}_{long_idx}.tif"
                )

                with rio.open(out_path, "w", **out_meta) as dest:
                    dest.write(out_image)
                files.append(out_path)
