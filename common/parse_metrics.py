from config import settings as st
from config import config as cfg
import pickle

import json
import os
import pandas as pd
from common import dictionary_utils


def _parse_metrics(file):
    """
    Save meaningfull metrics in csv format, such as "cls_accuracy", "false_negative", "fg_cls_accuracy", "loss_mask",
    "mask_rcnn/accuracy", "mask_rcnn/false_negative", "mask_rcnn/false_positive"

    Args:
        file: metrics,json

    Returns: None

    """
    iteration = []
    mask_rcnn_class_acc = []
    mask_rcnn_fg_accuracy = []
    mask_rcnn_mask_loss = []
    mask_rcnn_mask_accuracy = []
    mask_rcnn_mask_false_n = []
    mask_rcnn_false_p = []

    columns = ["Epochs","Mask R-CNN_Acc","Mask R-CNN_AccFG","Mask R-CNN_MaskLoss","Mask R-CNN_MaskAcc","Mask R-CNN_MaskFalseNegative","Mask R-CNN_MaskFalsePositive"]
    with open(file, "r") as f:
        metrics_list = [json.loads(line) for line in f]

    for index, metrics in enumerate(metrics_list):
        if index != len(metrics_list) - 1:
            iteration.append(metrics['iteration'])
            mask_rcnn_class_acc.append(metrics['fast_rcnn/cls_accuracy'])
            mask_rcnn_fg_accuracy.append(metrics['fast_rcnn/fg_cls_accuracy'])
            mask_rcnn_mask_loss.append(metrics['loss_mask'])
            mask_rcnn_mask_accuracy.append(metrics['mask_rcnn/accuracy'])
            mask_rcnn_mask_false_n.append(metrics['mask_rcnn/false_negative'])
            mask_rcnn_false_p.append(metrics['mask_rcnn/false_positive'])

    df = pd.DataFrame(list(
        zip(iteration, mask_rcnn_class_acc, mask_rcnn_fg_accuracy, mask_rcnn_mask_loss, mask_rcnn_mask_accuracy,
            mask_rcnn_mask_false_n, mask_rcnn_false_p)),
                      columns=["Epochs", "Mask R-CNN_Acc", "Mask R-CNN_AccFG", "Mask R-CNN_MaskLoss",
                               "Mask R-CNN_MaskAcc", "Mask R-CNN_MaskFalseNegative", "Mask R-CNN_MaskFalsePositive"])

    return df


if __name__ == '__main__':
    category = 'cat5'
    model_name = '101_fpn'
    model = 'output_142_128_R_101_FPN_6_v_Jul_Sep_resampling_factor0.001_0.0001_freeze_at2'
    excel_save = 'mask_rcnn_'+category+'_'+model_name
    folder = os.path.join(st.get_project_root(),'output',model)
    file = os.path.join(folder,'metrics.json')
    df = _parse_metrics(file)
    df.to_excel(os.path.join(folder,excel_save+'.xlsx'))



