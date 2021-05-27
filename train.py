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

import yaml


def register_data_set():
    train_path = os.path.join(settings.data_directory, config.year+'_processed', config._version_name,str(config.train_image_size),config._version_train_)
    validation_path = os.path.join(settings.data_directory,config.year+'_processed', config._version_name,str(config.train_image_size),config._version_validation_)
    register_coco_instances("veg_train_dataset", {}, os.path.join(train_path, 'annotation', 'train2020.json'),
                            os.path.join(train_path, 'images'))
    register_coco_instances("veg_val_dataset", {}, os.path.join(validation_path, 'annotation', 'val2020.json'),
                            os.path.join(validation_path, 'images'))

def calculate_num_classes(version_name):
    if version_name == 'v_Jan_Mar':
        return 4
    elif version_name == 'v_Apr_Jun':
        return 7
    elif version_name == 'v_Jul_Sep':
        return 7
    elif version_name == 'v_Oct_Dec':
        return 6
    else:
        return 2

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
    cfg.TEST.EVAL_PERIOD = 1000
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = os.path.join(settings.weights_directory, "model_final.pth")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.CHECKPOINT_PERIOD = 50000
    cfg.SOLVER.BASE_LR = config.learning_rate  # pick a good LR
    cfg.SOLVER.MAX_ITER = config.epochs
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = calculate_num_classes(config._version_name)
    cfg.DATALOADER.SAMPLER_TRAIN = 'RepeatFactorTrainingSampler'
    cfg.SOLVER.STEPS = (5000,)
    # cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    # cfg.OUTPUT_DIR = settings.check_point_output_directory
    cfg.OUTPUT_DIR = settings.data_directory +'/output'

    cfg.DATALOADER.NUM_WORKERS = 0 # for debug purposes

    if config.experiment_name == 'resampling_factor' :
        cfg.DATALOADER.REPEAT_THRESHOLD = config.experiment_value

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
    trainer = bt(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()




if __name__ == "__main__":
    """
    mention number of epochs in command line argument 'epochs'
    """
    main()
