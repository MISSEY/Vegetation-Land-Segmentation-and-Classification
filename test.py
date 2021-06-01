from detectron2.data.datasets import register_coco_instances

import os

from detectron2.config import get_cfg

from segmentation_model.train_net import BasePredictor as bp

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetCatalog, build_detection_train_loader, MetadataCatalog
from detectron2.data.build import filter_images_with_only_crowd_annotations
from detectron2.data import detection_utils as utils
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

import cv2

from config import settings
from config import config
import random


def inference_on_image(basepredictor,cfg,metadata,path):
    try:
        os.makedirs(os.path.join(config._test_output_image_path, config._test__model_))
    except OSError:
        print("Creation of the directory %s failed")
    else:
        print("Creation of the directory %s Success")
    data = DatasetCatalog.get("veg_val_dataset")

    # filter empty
    data = filter_images_with_only_crowd_annotations(data)
    # train_data_loader = build_detection_train_loader(cfg)
    gt = cv2.imread(os.path.join(settings.get_project_root(), 'figures','gt_'+ str(config.test_image_size) +'.png'))
    inf = cv2.imread(os.path.join(settings.get_project_root(), 'figures','inf_'+ str(config.test_image_size) +'.png'))
    gt_inf = cv2.hconcat([gt, inf])
    for d in data:
        img = utils.read_image(d["file_name"], "RGB")
        im = cv2.imread(d["file_name"])

        outputs = basepredictor(im)

        v = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=1,
                       instance_mode=ColorMode.IMAGE_BW
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        visualizer = Visualizer(img, metadata=metadata, scale=1)
        gt = visualizer.draw_dataset_dict(d)

        ## Save ground_truth and inference

        gt = cv2.cvtColor(gt.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
        inf = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)

        file_name = os.path.basename(d["file_name"])
        file_save_path = os.path.join(path,file_name)
        # gt_path = os.path.join(path,file_name[0] + "_gt_."+file_name[1])
        # inf_path = os.path.join(path, file_name[0] + "_inf_." + file_name[1])

        # horizontal and vertical concat
        im_h = cv2.hconcat([gt, inf])
        im_v = cv2.vconcat([gt_inf, im_h])

        cv2.imwrite(file_save_path,im_v)


        

def main(inference = False):
    validation_path = os.path.join(settings.data_directory, config.test_year + '_processed', config._test_version_name,
                                   str(config.test_image_size), config._version_validation_)
    register_coco_instances("veg_val_dataset", {}, os.path.join(validation_path, 'annotation', 'val'+config.test_year+'.json'),
                            os.path.join(validation_path, 'images'))
    cfg = get_cfg()

    output_path = os.path.join('output', config._test__model_)
    cfg.merge_from_file(os.path.join(output_path, 'config.yaml'))
    cfg.DATASETS.TEST = ("veg_val_dataset",)
    # cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = os.path.join(output_path, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    # trainer = bt(cfg)
    predictor = bp(cfg)
    metadata = MetadataCatalog.get("veg_val_dataset")

    if inference:
        outputpath = os.path.join(config._test_output_image_path,config._test__model_)
        inference_on_image(predictor,cfg,metadata,path = outputpath)

    else:
        evaluator = COCOEvaluator("veg_val_dataset", ("bbox", "segm"), False, output_dir=output_path)
        val_loader = build_detection_test_loader(cfg, "veg_val_dataset")
        inference_stats = inference_on_dataset(predictor.model, val_loader, evaluator)
        print(inference_stats)




if __name__ == '__main__':
    main(inference= True)
