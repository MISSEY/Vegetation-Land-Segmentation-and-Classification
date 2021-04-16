# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

# Directories names
_data_ = 'Data'

# Below directories should be inside _data_ , otherwise give full path instead of names

_raw_shape_directory = 'Raw_shape_files'
_processed_root_directory_ = 'processed_data'
_processed_denmark_root_ = 'processed_denmark_files'
_version_ = 'v_3'
_denmark_tif_ = 'denmark_tif'
_denmark_shape_ = 'denmark_shapes_shp'
_train_ = 'train'
_validation_ = 'validation'
_crop_images_ = 'cropped_images'
_chip_dfs_ = 'chip_dfs_pickles'

_filter_train_ = 'filter_train'
_filter_validation_= 'filter_validation'


# File names
__denmark_street_shape = 'denmark_vegetation/'


# Image
# Image configuration depend upon the data version above v_3 : 800 x 800, v_5 : 400 x 400, v_6 : 200 x 200

image_size = 400

instance_per_image_filter = 2000

# Training

epochs = 60000