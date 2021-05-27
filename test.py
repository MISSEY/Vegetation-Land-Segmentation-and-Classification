from detectron2.data.datasets import register_coco_instances

import os

from detectron2.config import get_cfg

from segmentation_model.train_net import BasePredictor as bp

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from config import settings
from config import config


def main():
    validation_path = os.path.join(settings.data_directory, config.year + '_processed', config._version_name,
                                   str(config.train_image_size), config._version_validation_)
    register_coco_instances("veg_val_dataset", {}, os.path.join(validation_path, 'annotation', 'val2020.json'),
                            os.path.join(validation_path, 'images'))
    cfg = get_cfg()

    output_path = os.path.join('output', config._test__model_)
    cfg.merge_from_file(os.path.join(output_path, 'config.yaml'))
    cfg.DATASETS.TEST = ("veg_val_dataset",)
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = os.path.join(output_path, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    # trainer = bt(cfg)
    predictor = bp(cfg)

    evaluator = COCOEvaluator("veg_val_dataset", ("bbox", "segm"), False, output_dir=output_path)
    val_loader = build_detection_test_loader(cfg, "veg_val_dataset")
    inference_stats = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(inference_stats)


if __name__ == '__main__':
    main()
