"""
Change settings here
"""

from pathlib import Path
from config import config
import os

def get_project_root()-> Path:
    return Path(__file__).parent.parent



# while training on cluster
check_point_output_directory = '/netscratch/smishra/output_400'


data_directory = os.path.join(get_project_root(), 'Data')

# mention the raw shape file name in the data directory
shape_file = os.path.join(data_directory, config.__denmark_street_shape)

version = os.path.join(data_directory, config._version_)
processed_data_directory: str = os.path.join(version, config._processed_root_directory_)


# Mention the raster file extracted from the google earth engine
denmark_tif_directory = os.path.join(data_directory, config._processed_denmark_root_, config._denmark_tif_)

# Mention the shape files generated for each tif extracted from the google earth engine
denmark_shp_directory = os.path.join(data_directory, config._processed_denmark_root_, config._denmark_shape_)

# Mention the file where after the chips dictionary will get saved
chips_df_dictionary_directory = os.path.join(processed_data_directory, config._chip_dfs_)

# Mention the folder name where the cropped images will be saved
crop_images_output_path = os.path.join(processed_data_directory, config._crop_images_)

# Mention the train and validation directories
train = os.path.join(version, config._train_)
validation = os.path.join(version, config._validation_)
