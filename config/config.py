# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

# debug
debug = False

# Directories names
_data_ = 'Data'

# Below directories should be inside _data_ , otherwise give full path instead of names
_raw_shape_directory = 'Raw_shape_files'
_tif_ = 'Denmark_tif'
_version_ = str(6)
_version_name ='v_whole_summer_winter_2020_singleclass'
_version_processed_shape_files = 'processed_shape_files' # Each version has unique files depending on category. ,
# It must be inside version name
_version_train_ = 'train'
_version_validation_ = 'validation'
_version_crop_images_ = 'cropped_images'

# train
train_config = {
    'train_year' : '2020',
    # change version name while training
    'train_image_size' : 128,
    'instance_per_image_filter' : 2000,
    "epochs" : 20000,
    "experiment_name" : 'resampling_factorNo',
    "experiment_value" : 0.001,  # from lvis paper
    "freeze_at" : 0,
    "validation" : True,
    "learning_rate" : 0.0001,
    # model
    "FPN" : False,
    "PIXEL_MEAN" : [49.21536236,95.77637493,98.01276427],
    "PIXEL_STD" : [9.454635453006942, 11.255133601289831, 15.095056038238923],
    "model_name" : 'R_50_FPN', # R_101_X_FPN, R_101_FPN, R_50_FPN
    "backbone_name" : 'build_resnet_fpn_backbone', # build_resnet_fpn_backbone (default) # build_resnet_backbone_custom
    "architecture_name" : "GeneralizedRCNN",  # for FCIS implementation (default) #GeneralizedRCNN,    GeneralizedFCIS
    "config_file" : "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", # mask_rcnn_X_101_32x8d_FPN_3x , mask_rcnn_R_101_FPN_3x , mask_rcnn_R_50_FPN_3x
    "batch_size" : 2048,
    "experiment_number" : 301,
    "checkpoint_period" : 10000,
    "eval_period" : 5000,
    "solver_steps" : (2000,5000,15000),
    "train_from_scratch" : False,
    "MODEL.RPN.PRE_NMS_TOPK_TRAIN" : 12000,
    "MODEL.RPN.PRE_NMS_TOPK_TEST" : 6000,
    "SOLVER.WARMUP_ITERS" : 4000
}
fcis_model = {
    'flag':False,
    'RPN_NMS_THRESH' : 0.7,
    'RPN_IN_FEATURES' : ['res5'],
    "backbone_name" : 'build_resnet_backbone_custom',
    "architecture_name": "FCISProposalNetwork", # GeneralizedFCIS, #FCISProposalNetwork
    "ANCHOR_GENERATOR.SIZES" : [[32]],
    "MODEL.RESNETS.NORM" : "BN",
    "MODEL.ROI_BOX_HEAD.POOLER_TYPE":"ROIPool",

}

# raster image
crs = "EPSG:32632"

# testing
_test_version_name = 'v_whole_summer_winter_2020'
_test__model_ = 'output_56_800_R_50_FPN_v_whole_summer_winter_2020_resampling_factor0.01_0.001_freeze_at2'
_test_output_image_path = 'output_image'
test_year = '2020'
test_image_size = 800
test_version_ = 1

# generate data
generate_year = '2020'
data_generation_image_size = 128
denmark_shape_directory = 'Denmark_shape_2020'
_preprocessed_denamark_shape_files_ = 'Preprocessed_denamark_shape_files'

