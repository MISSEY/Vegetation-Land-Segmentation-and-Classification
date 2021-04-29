import torch, torchvision
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from segmentation_model.visualizer import Basevisualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
import os
import cv2
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
import IPython
from common import dictionary_utils
from common import coco_utils
from segmentation_model.train_net import BasePredictor as bp

def main():
    register_coco_instances("street_val_dataset", {},
                            os.path.join('Data/v_3/filter_validation', 'annotation', 'val2020.json'),
                            os.path.join('Data/v_3/filter_validation/filter_validation', 'images'))
    cfg = get_cfg()

    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    # cfg.MODEL.DEVICE = "cpu"
    # predictor = bp(cfg)

    cfg.merge_from_file('outputs/output_800_R_50_FPN/config.yaml')
    cfg.DATASETS.TEST = ("street_val_dataset",)
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = os.path.join('outputs/output_800_R_50_FPN', "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    predictor = bp(cfg)

    im = cv2.imread("Data/v_3/filter_validation/images/COCO_val2021_000000100741.jpg")
    # im = cv2.imread("000000439715.jpg")
    outputs = predictor(im)



if __name__ == '__main__':
    main()