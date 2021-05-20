# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

# Directories names
_data_ = 'Data'

# Below directories should be inside _data_ , otherwise give full path instead of names

_raw_shape_directory = 'Raw_shape_files'
_preprocessed_denamark_shape_files_ = 'Preprocessed_denamark_shape_files'


# change version name while training
_version_name ='v_Jan_Mar'

_version_train_ = 'train'
_version_validation_ = 'validation'
_version_crop_images_ = 'cropped_images'


# Image
# Image configuration depend upon the data version above v_3 : 800 x 800, v_5 : 400 x 400, v_6 : 200 x 200

train_image_size = 400
test_image_size = 800

instance_per_image_filter = 2000

# Training

epochs = 100000

# experiment

experiment_name = 'negative_sampling'
experiment_value = 0  # from lvis paper

validation = True

learning_rate = 0.001

# model
model_name = 'R_50_FPN'




