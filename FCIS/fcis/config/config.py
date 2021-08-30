# --------------------------------------------------------
# Fully Convolutional Instance-aware Semantic Segmentation
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Haozhi Qi
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

import yaml
import numpy as np
from easydict import EasyDict as edict

config = edict()

config.MXNET_VERSION = ''
config.output_path = ''
config.symbol = ''
config.gpus = ''
config.CLASS_AGNOSTIC = True
config.SCALES = [(600, 1000)]  # first is scale (the shorter side); second is max size
config.MASK_SIZE = 21
config.BINARY_THRESH = 0.4

# default training
config.default = edict()
config.default.frequent = 20
config.default.kvstore = 'device'

# network related params
config.network = edict()
config.network.pretrained = './model/pretrained_model/resnet_v2_101'
config.network.pretrained_epoch = 0
config.network.PIXEL_MEANS = np.array([0, 0, 0])
config.network.IMAGE_STRIDE = 0
config.network.RPN_FEAT_STRIDE = 16
config.network.RCNN_FEAT_STRIDE = 16
config.network.FIXED_PARAMS = ['conv0', 'stage1', 'gamma', 'beta']
config.network.FIXED_PARAMS_SHARED = ['conv0', 'stage1', 'stage2', 'stage3', 'gamma', 'beta']
config.network.ANCHOR_SCALES = (8, 16, 32)
config.network.ANCHOR_RATIOS = (0.5, 1, 2)
config.network.NUM_ANCHORS = len(config.network.ANCHOR_SCALES) * len(config.network.ANCHOR_RATIOS)

# dataset related params
config.dataset = edict()
config.dataset.dataset = 'coco'
config.dataset.image_set = 'SDS_train'
config.dataset.test_image_set = 'SDS_val'
config.dataset.root_path = './data'
config.dataset.dataset_path = './data/VOCdevkit'
config.dataset.NUM_CLASSES = 5

# Training configurations
config.TRAIN = edict()

config.TRAIN.lr = 0.0005
config.TRAIN.lr_step = ''
config.TRAIN.warmup = False
config.TRAIN.warmup_lr = 0
config.TRAIN.warmup_step = 0
config.TRAIN.momentum = 0.9
config.TRAIN.wd = 0.0005
config.TRAIN.begin_epoch = 0
config.TRAIN.end_epoch = 0
config.TRAIN.model_prefix = ''

# whether resume training
config.TRAIN.RESUME = False
# whether flip image
config.TRAIN.FLIP = True
# whether shuffle image
config.TRAIN.SHUFFLE = True
# whether use OHEM
config.TRAIN.ENABLE_OHEM = False
# size of images for each device, 2 for rcnn, 1 for rpn and e2e
config.TRAIN.BATCH_IMAGES = 2
# e2e changes behavior of anchor loader and metric
config.TRAIN.END2END = False
# group images with similar aspect ratio
config.TRAIN.ASPECT_GROUPING = True

# R-CNN
# rcnn rois batch size
config.TRAIN.BATCH_ROIS = 128
config.TRAIN.BATCH_ROIS_OHEM = 128
# rcnn rois sampling params
config.TRAIN.FG_FRACTION = 0.25
config.TRAIN.FG_THRESH = 0.5
config.TRAIN.BG_THRESH_HI = 0.5
config.TRAIN.BG_THRESH_LO = 0.0
# rcnn bounding box regression params
config.TRAIN.BBOX_REGRESSION_THRESH = 0.5
config.TRAIN.BBOX_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0])

# RPN anchor loader
# rpn anchors batch size
config.TRAIN.RPN_BATCH_SIZE = 256
# rpn anchors sampling params
config.TRAIN.RPN_FG_FRACTION = 0.5
config.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
config.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
config.TRAIN.RPN_CLOBBER_POSITIVES = False
# rpn bounding box regression params
config.TRAIN.RPN_BBOX_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
config.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

# used for end2end training
# RPN proposal
config.TRAIN.CXX_PROPOSAL = True
config.TRAIN.RPN_NMS_THRESH = 0.7
config.TRAIN.RPN_PRE_NMS_TOP_N = 12000
config.TRAIN.RPN_POST_NMS_TOP_N = 2000
config.TRAIN.RPN_MIN_SIZE = config.network.RPN_FEAT_STRIDE
config.TRAIN.RPN_ALLOWED_BORDER = 0
# whether select from all rois or bg rois
config.TRAIN.GAP_SELECT_FROM_ALL = True
config.TRAIN.IGNORE_GAP = False
# binary_threshold for proposal annotator
config.TRAIN.BINARY_THRESH = 0.4
# approximate bounding box regression
config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = False
config.TRAIN.BBOX_MEANS = (0.0, 0.0, 0.0, 0.0)
config.TRAIN.BBOX_STDS = (0.1, 0.1, 0.2, 0.2)
# loss weight for cls, bbox and mask
config.TRAIN.LOSS_WEIGHT = (1.0, 10.0, 1.0)
config.TRAIN.CONVNEW3 = False


config.TEST = edict()

# R-CNN testing
# use rpn to generate proposal
config.TEST.HAS_RPN = False
# size of images for each device
config.TEST.BATCH_IMAGES = 1

# RPN proposal
config.TEST.CXX_PROPOSAL = True
config.TEST.RPN_NMS_THRESH = 0.7
config.TEST.RPN_PRE_NMS_TOP_N = 6000
config.TEST.RPN_POST_NMS_TOP_N = 300
config.TEST.RPN_MIN_SIZE = config.network.RPN_FEAT_STRIDE

# RPN generate proposal
config.TEST.PROPOSAL_NMS_THRESH = 0.7
config.TEST.PROPOSAL_PRE_NMS_TOP_N = 20000
config.TEST.PROPOSAL_POST_NMS_TOP_N = 2000
config.TEST.PROPOSAL_MIN_SIZE = config.network.RPN_FEAT_STRIDE

# RCNN nms
config.TEST.NMS = 0.3

# Test Model Epoch
config.TEST.test_epoch = 0

# TEST iteration
config.TEST.ITER = 1
config.TEST.MIN_DROP_SIZE = 16

# mask merge
config.TEST.USE_MASK_MERGE = False
config.TEST.USE_GPU_MASK_MERGE = False
config.TEST.MASK_MERGE_THRESH = 0.5

# version_name_and _year
config.versionnameandyear = 'whole_summer_winter_2020'
config.image_size = '128'

#demo
config.numclasses = 5
config.ownclasses = ["barley_w","wheat_w","rye_w","rapeseed_w"]
config.imagenames = ['COCO_val2020_whole_summer_winter_2020_singleclass_000000102308.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000102313.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000102325.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000102326.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000102327.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000109270.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000109288.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000109291.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000109307.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000109543.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000109547.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000109567.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000109583.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000109585.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000109597.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000110352.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000110360.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000110367.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000110369.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000110377.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000110384.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000110392.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000110407.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000102331.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000109594.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000111238.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000112023.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000113953.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000111241.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000111243.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000111247.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000111477.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000111495.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000111514.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000111516.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000112006.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000112009.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000112043.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000112053.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000112060.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000112061.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000113695.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000113705.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000113707.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000113950.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000113951.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000113958.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000113959.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000113976.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000113980.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000114202.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000115313.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000115322.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000115325.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000115327.jpg',
 'COCO_val2020_whole_summer_winter_2020_singleclass_000000115343.jpg']
config.modelprefix ='/netscratch/smishra/thesis/output/fcis/v_Jan_Mar/128/resnet_v1_101_coco_fcis_end2end_ohem/train2020/e2e'
config.testepoch = 7


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    if k == 'TRAIN':
                        if 'BBOX_WEIGHTS' in v:
                            v['BBOX_WEIGHTS'] = np.array(v['BBOX_WEIGHTS'])
                    elif k == 'network':
                        if 'PIXEL_MEANS' in v:
                            v['PIXEL_MEANS'] = np.array(v['PIXEL_MEANS'])
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("key must exist in config.py")
