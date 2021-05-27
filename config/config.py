# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

# Directories names
_data_ = 'Data'

# Below directories should be inside _data_ , otherwise give full path instead of names

_raw_shape_directory = 'Raw_shape_files'
_tif_ = 'Denmark_tif'


# Year of data
year = '2019'

# change version name while training
_version_name ='v_Jan_Mar'

_version_processed_shape_files = 'processed_shape_files' # Each version has unique files depending on category. ,
# It must be inside version name

_version_train_ = 'train'
_version_validation_ = 'validation'
_version_crop_images_ = 'cropped_images'


# Image
# Image configuration depend upon the data version above v_3 : 800 x 800, v_5 : 400 x 400, v_6 : 200 x 200

data_generation_image_size = 800
train_image_size = 400
test_image_size = 400
instance_per_image_filter = 2000

# Training

epochs = 100000

experiment_name = 'resampling_factor'
experiment_value = 0.001  # from lvis paper

validation = True

learning_rate = 0.001

# model
model_name = 'R_50_DC5'

# raster image
crs = "EPSG:4326"

# testing
_test_version_name = 'v_Jan_Mar'
_test__model_ = 'output_400_R_50_FPN_v_Apr_Jun_merge_training0_0.001_33'


