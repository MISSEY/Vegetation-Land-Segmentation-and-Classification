# Initialize the detectron2 logger and set its verbosity level to “DEBUG”.

from detectron2.utils.logger import setup_logger

setup_logger()

import os

from detectron2 import model_zoo

from detectron2.config import get_cfg

from config import settings, config
from common import dictionary_utils

from detectron2.data.datasets import register_coco_instances

from segmentation_model.train_net import BaseTrainer as bt

import yaml





def get_path():
    """
    Get path for train and validation dataset

    :return:
    """

    # for debugging
    if (config.debug):

        train_path = os.path.join(settings.data_directory, str(config._version_),
                                  config.train_config["train_year"] + '_processed', config._version_name,
                                  str(config.train_config["train_image_size"]), config._version_train_)
        validation_path = os.path.join(settings.data_directory, str(config._version_),
                                       config.train_config["train_year"] + '_processed', config._version_name,
                                       str(config.train_config["train_image_size"]), config._version_validation_
                                       )
    else:
        train_path = os.path.join(settings.data_directory_cluster,
                                  str(config._version_),
                                  config.train_config["train_year"] + '_processed',
                                  config._version_name,
                                  str(config.train_config["train_image_size"]),
                                  config._version_train_
                                  )
        validation_path = os.path.join(settings.data_directory_cluster,
                                       str(config._version_),
                                       config.train_config["train_year"] + '_processed',
                                       config._version_name,
                                       str(config.train_config["train_image_size"]),
                                       config._version_validation_
                                       )
    return train_path,validation_path


def register_data_set():
    """
    Register coco dataset on Detectron2
    Returns:

    """
    train_path,validation_path = get_path()

    register_coco_instances("veg_train_dataset", {},
                            os.path.join(train_path, 'annotation', 'train' + config.train_config["train_year"] + '.json'),
                            os.path.join(train_path, 'images'))
    register_coco_instances("veg_val_dataset", {},
                            os.path.join(validation_path, 'annotation', 'val' + config.train_config["train_year"] + '.json'),
                            os.path.join(validation_path, 'images'))


def calculate_num_classes(version_name):
    train_path, validation_path = get_path()
    annon = dictionary_utils.load_json(os.path.join(train_path, 'annotation', 'train' + config.train_config["train_year"] + '.json'))
    classes = len(annon['categories'])
    return(classes)


def setup():
    """
    Create configs and perform basic setups.
    """

    register_data_set()

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config.train_config["config_file"]))
    # cfg.merge_from_file(os.path.join("config.yaml"))
    cfg.DATASETS.TRAIN = ("veg_train_dataset",)
    # cfg.DATASETS.TRAIN = ("street_val_dataset",)
    cfg.DATASETS.TEST = ("veg_val_dataset",)
    # cfg.DATASETS.TEST = ()
    cfg.TEST.EVAL_PERIOD = config.train_config["eval_period"]
    # cfg.MODEL.WEIGHTS = os.path.join(settings.weights_directory, "model_final.pth")
    cfg.SOLVER.CHECKPOINT_PERIOD = config.train_config["checkpoint_period"]
    cfg.SOLVER.BASE_LR = config.train_config["learning_rate"]
    cfg.SOLVER.MAX_ITER = config.train_config["epochs"]
    # cfg.INPUT.MASK_FORMAT = "polygon"
    # cfg.MODEL.RPN.NMS_THRESH = 0.7
    # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = calculate_num_classes(config._version_name)
    cfg.SOLVER.STEPS = config.train_config["solver_steps"]
    # cfg.INPUT.MIN_SIZE_TRAIN = (800,)

    # To stop auto resize
    cfg.INPUT.MIN_SIZE_TEST = 0

    cfg.MODEL.PIXEL_MEAN = config.train_config["PIXEL_MEAN"]
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = config.train_config["MODEL.RPN.PRE_NMS_TOPK_TRAIN"]
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = config.train_config["MODEL.RPN.PRE_NMS_TOPK_TEST"]
    cfg.SOLVER.WARMUP_ITERS = config.train_config["SOLVER.WARMUP_ITERS"]
    cfg.TEST.DETECTIONS_PER_IMAGE = 200
    # cfg.MODEL.PIXEL_STD = config.train_config["PIXEL_STD"]

    if(config.train_config["FPN"]):
        cfg.MODEL.BACKBONE.NAME = config.train_config["backbone_name"]
        cfg.MODEL.META_ARCHITECTURE = config.train_config["architecture_name"]

    cfg.MODEL.BACKBONE.FREEZE_AT = config.train_config["freeze_at"]

    if config.debug:
        cfg.DATALOADER.NUM_WORKERS = 0  # for debug purposes
        cfg.OUTPUT_DIR = settings.data_directory + '/output'
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1
        cfg.SOLVER.IMS_PER_BATCH = 1
    else:
        cfg.OUTPUT_DIR = settings.check_point_output_directory
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config.train_config["batch_size"]
        cfg.SOLVER.IMS_PER_BATCH = 2

    if config.train_config["experiment_name"] == 'resampling_factor':
        cfg.DATALOADER.SAMPLER_TRAIN = 'RepeatFactorTrainingSampler'
        cfg.DATALOADER.REPEAT_THRESHOLD = config.train_config["experiment_value"]

    if not config.train_config["train_from_scratch"]:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config.train_config["config_file"])  # Let training initialize from model zoo
        cfg.MODEL.BACKBONE.FREEZE_AT = 0
    else:
        # scratch training
        if not config.fcis_model['flag']:
            cfg.MODEL.WEIGHTS = ''

    # cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    if config.fcis_model['flag']:
        cfg.MODEL.BACKBONE.NAME = config.fcis_model["backbone_name"]
        cfg.MODEL.META_ARCHITECTURE = config.fcis_model["architecture_name"]
        cfg.MODEL.RPN.IN_FEATURES = config.fcis_model["RPN_IN_FEATURES"]
        cfg.MODEL.ANCHOR_GENERATOR.SIZES =config.fcis_model["ANCHOR_GENERATOR.SIZES"]
        cfg.MODEL.RESNETS.NORM = config.fcis_model["MODEL.RESNETS.NORM"]
        cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = config.fcis_model["MODEL.ROI_BOX_HEAD.POOLER_TYPE"]


    return cfg


def save_config_yaml(cfg):
    dict_ = yaml.safe_load(cfg.dump())
    with open(os.path.join(cfg.OUTPUT_DIR, 'config.yaml'), 'w') as file:
        _ = yaml.dump(dict_, file)


def main():
    cfg = setup()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    save_config_yaml(cfg)
    trainer = bt(cfg,config.train_config)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    """
    mention number of epochs in command line argument 'epochs'
    """
    main()
