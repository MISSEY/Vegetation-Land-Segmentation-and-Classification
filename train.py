# Initialize the detectron2 logger and set its verbosity level to “DEBUG”.

from detectron2.utils.logger import setup_logger

setup_logger()

from shutil import copyfile

import os

from detectron2 import model_zoo

from detectron2.config import get_cfg

from config import settings, config
from common import dictionary_utils

from detectron2.data.datasets import register_coco_instances

from detectron2.engine import default_argument_parser, default_setup, launch
from segmentation_model.train_net import BaseTrainer as bt
from backbone import resnet

import yaml

def get_path():
    """

    :return:
    """
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
    train_path,validation_path = get_path()

    register_coco_instances("veg_train_dataset", {},
                            os.path.join(train_path, 'annotation', 'train' + config.train_config["train_year"] + '.json'),
                            os.path.join(train_path, 'images'))
    register_coco_instances("veg_val_dataset", {},
                            os.path.join(validation_path, 'annotation', 'val' + config.train_config["train_year"] + '.json'),
                            os.path.join(validation_path, 'images'))


def calculate_num_classes(version_name):
    train_path, validation_path = get_path()
    annon = dictionary_utils.load_json(os.path.join(validation_path, 'annotation', 'val' + config.train_config["train_year"] + '.json'))
    classes = len(annon['categories'])
    return(classes)


def setup():
    """
    Create configs and perform basic setups.
    """

    register_data_set()

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.merge_from_file(os.path.join(settings.weights_directory, "config.yaml"))
    cfg.DATASETS.TRAIN = ("veg_train_dataset",)
    # cfg.DATASETS.TRAIN = ("street_val_dataset",)
    cfg.DATASETS.TEST = ("veg_val_dataset",)
    # cfg.DATASETS.TEST = ()
    cfg.TEST.EVAL_PERIOD = config.train_config["eval_period"]
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = os.path.join(settings.weights_directory, "model_final.pth")
    cfg.SOLVER.CHECKPOINT_PERIOD = config.train_config["checkpoint_period"]
    cfg.SOLVER.BASE_LR = config.train_config["learning_rate"]  # pick a good LR
    cfg.SOLVER.MAX_ITER = config.train_config["epochs"]

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = calculate_num_classes(config._version_name)
    cfg.DATALOADER.SAMPLER_TRAIN = 'RepeatFactorTrainingSampler'
    cfg.SOLVER.STEPS = config.train_config["solver_steps"]
    # cfg.INPUT.MIN_SIZE_TRAIN = (800,)

    cfg.INPUT.MIN_SIZE_TEST = 0

    cfg.MODEL.BACKBONE.NAME = config.train_config["backbone_name"]
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
        cfg.DATALOADER.REPEAT_THRESHOLD = config.train_config["experiment_value"]

    # cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

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
