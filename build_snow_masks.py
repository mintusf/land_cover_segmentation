import glob
import os
from shutil import copyfile
import rasterio as rio
import numpy as np
from rasterio.mask import mask

from utils.utilities import get_raster_filepath

search_dir = "/data/seg_data/snow"
ROI = "ROIs2020_snow"
modes = ["train", "val", "test"]

def build_snow_mask(ref_img: str, savepath: str):
    """Convert np.array to raster and save
    Args:
        img_np (np.array): Image to be saved
        ref_img (str): Referenced raster
        savepath (str): Output raster savepath (tif format is recommended)
    """
    with rio.open(ref_img) as src:
        transform = src.transform
        size = (src.height, src.width)

    mask = np.zeros([4,size[0],size[0]])
    mask[3,:,:] = 2
    os.makedirs(os.path.split(savepath)[0], exist_ok=True)

    with rio.open(
        savepath,
        "w",
        driver="GTiff",
        dtype=mask.dtype,
        height=size[0],
        width=size[1],
        count=4,
        crs=src.crs,
        transform=transform,
    ) as dst:
        dst.write(mask)

samples = []
for file in glob.glob(f"{search_dir}/{ROI}/*/*tif"):
    filename = os.path.splitext(os.path.split(file)[1])[0]
    if "lc" in filename:
        break
    print(file)
    sample_name = filename.replace("s2_","")
    samples.append(sample_name)

    img_path = get_raster_filepath(search_dir, sample_name, sensor="s2")
    mask_path = get_raster_filepath(search_dir, sample_name, sensor="lc")

    # copyfile(file, img_path)
    build_snow_mask(img_path, mask_path)

with open("/data/seg_data/snow/list.txt", "w") as fp:
    for sample in samples:
        fp.write(sample + "\n")