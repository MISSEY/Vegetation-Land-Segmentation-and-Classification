# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

# debug
debug = True

# Directories names
_data_ = 'Data'

# Below directories should be inside _data_ , otherwise give full path instead of names

_raw_shape_directory = 'Raw_shape_files'
_tif_ = 'Denmark_tif'


# train
train_year = '2020'
# change version name while training
_version_name ='v_whole_summer_winter_2020'
_version_processed_shape_files = 'processed_shape_files' # Each version has unique files depending on category. ,
# It must be inside version name
_version_train_ = 'train'
_version_validation_ = 'validation'
_version_crop_images_ = 'cropped_images'
train_image_size = 800
instance_per_image_filter = 2000

epochs = 100000
experiment_name = 'resampling_factor'
experiment_value = 0.001  # from lvis paper
freeze_at = 0
validation = True
learning_rate = 0.00025
# model
model_name = 'R_50_FPN'
backbone_name ='build_resnet_fpn_backbone' # build_resnet_fpn_backbone (default) # build_resnet_backbone_custom

batch_size = 128

# raster image
crs = "EPSG:4326"

# testing
_test_version_name = 'v_whole_summer_winter_2020'
_test__model_ = 'output_800_R_50_FPN_v_whole_summer_winter_2020_resampling_factor0.001_0.001_freeze_at0_54'
_test_output_image_path = 'output_image'
test_year = '2020'
test_image_size = 800

# generate data
generate_year = '2020'
data_generation_image_size = 128
denmark_shape_directory = 'Denmark_shape_2020'
_preprocessed_denamark_shape_files_ = 'Preprocessed_denamark_shape_files'

