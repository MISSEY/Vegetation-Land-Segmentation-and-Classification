from config import settings as st
from typing import Tuple, Generator, List, Union
from rasterio.windows import Window
from shapely.geometry import Polygon

import affine
import rasterio
from tqdm import tqdm
import numpy as np
from skimage import exposure, img_as_ubyte
import itertools
import shapely
import warnings
from PIL import Image as pilimg
import os

def get_chip_windows(raster_width,raster_height,raster_transform,chip_width,chip_height,skip_partial_chips = False) -> Generator[Tuple[Window, affine.Affine, Polygon], any, None]:
    """
    Generate windows of defined width and height for vectors
    :param raster_width: width of raster image
    :param raster_height: height of raster image
    :param raster_transform: transformation of raster image
    :param chip_width: output width of each chips
    :param chip_height: output height of each chips
    :param skip_partial_chips: Ignore the edge chips that are not in full size
    :return: Yields the tuple of chip window, transform and polygon
    """


    col_row_offsets = itertools.product(range(0, raster_width, chip_width), range(0, raster_height, chip_height))
    raster_window = Window(col_off=0, row_off=0, width=raster_width, height=raster_height)

    for col_off, row_off in col_row_offsets:
        chip_window = Window(col_off=col_off, row_off=row_off, width=chip_width, height=chip_height)

        if skip_partial_chips:
            if row_off + chip_height > raster_height or col_off + chip_width > raster_width:
                continue

        chip_window = chip_window.intersection(raster_window)
        chip_transform = rasterio.windows.transform(chip_window, raster_transform)
        chip_bounds = rasterio.windows.bounds(chip_window, raster_transform)  # Uses transform of full raster.
        chip_poly = shapely.geometry.box(*chip_bounds, ccw=False)

        yield (chip_window, chip_transform, chip_poly)

def cut_chip_images(inpath_raster,
                    outpath_chipfolder,
                    chip_names,
                    chip_windows,
                    raster_image_range,
                    bands=[3, 2, 1]):
    """Cuts image raster to chips via the given windows and exports them to jpg."""

    src = rasterio.open(inpath_raster)

    all_chip_stats = {}
    for chip_name, chip_window in tqdm(zip(chip_names, chip_windows)):
        img_array = np.dstack(list(src.read(window=chip_window)))
        #img_array = exposure.rescale_intensity(img_array, in_range=(0, raster_image_range))  # Sentinel2 range.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img_array = img_as_ubyte(img_array)
        img_pil = pilimg.fromarray(img_array)

        # Export chip images
        dst = os.path.join(outpath_chipfolder,str(chip_name) + '.jpg')
        img_pil.save(dst, format='JPEG', subsampling=0, quality=100)

        all_chip_stats[chip_name] = {'mean': img_array.mean(axis=(0, 1)),
                                     'std': img_array.std(axis=(0, 1))}
    src.close()

    return all_chip_stats



