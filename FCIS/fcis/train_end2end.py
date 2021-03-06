# --------------------------------------------------------
# Fully Convolutional Instance-aware Semantic Segmentation
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Guodong Zhang, Haozhi Qi
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------


import _init_paths

import argparse
import cv2
import pprint
import os
import sys
from config.config import config, update_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train FCIS Network')
    # general
    # configuration file is required
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent', help='frequency of logging', default=config.default.frequent, type=int)
    args = parser.parse_args()
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))

import shutil
import mxnet as mx
import numpy as np

from symbols import *
from core import callback, metric
from core.loader import AnchorLoader
from core.module import MutableModule
from utils.create_logger import create_logger
from utils.load_data import load_gt_sdsdb, merge_roidb, filter_roidb
from utils.load_model import load_param
from utils.PrefetchingIter import PrefetchingIter
from utils.lr_scheduler import WarmupMultiFactorScheduler


def train_net(args, ctx, pretrained, epoch, prefix, begin_epoch, end_epoch, lr, lr_step):
    mx.random.seed(3)
    np.random.seed(3)

    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)
    prefix = os.path.join(final_output_path, prefix)

    # load symbol
    shutil.copy2(os.path.join(curr_path, 'symbols', config.symbol + '.py'), final_output_path)
    sym_instance = eval(config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=True)
    feat_sym = sym.get_internals()['rpn_cls_score_output']

    # setup multi-gpu
    batch_size = len(ctx)
    input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size

    # print config
    pprint.pprint(config)
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    # load dataset and prepare imdb for training
    image_sets = [iset for iset in config.dataset.image_set.split('+')]
    sdsdbs = [load_gt_sdsdb(config.dataset.dataset, image_set, config.dataset.root_path, config.dataset.dataset_path,
                            config.versionnameandyear,mask_size=config.MASK_SIZE, binary_thresh=config.BINARY_THRESH,
                            result_path=final_output_path, flip=config.TRAIN.FLIP)
              for image_set in image_sets]
    sdsdb = merge_roidb(sdsdbs)
    sdsdb = filter_roidb(sdsdb, config)

    # load training data
    train_data = AnchorLoader(feat_sym, sdsdb, config, batch_size=input_batch_size, shuffle=config.TRAIN.SHUFFLE,
                              ctx=ctx, feat_stride=config.network.RPN_FEAT_STRIDE, anchor_scales=config.network.ANCHOR_SCALES,
                              anchor_ratios=config.network.ANCHOR_RATIOS, aspect_grouping=config.TRAIN.ASPECT_GROUPING,
                              allowed_border=config.TRAIN.RPN_ALLOWED_BORDER)

    # infer max shape
    max_data_shape = [('data', (config.TRAIN.BATCH_IMAGES, 3,
                                max([v[0] for v in config.SCALES]), max(v[1] for v in config.SCALES)))]
    max_data_shape, max_label_shape = train_data.infer_shape(max_data_shape)
    max_data_shape.append(('gt_boxes', (config.TRAIN.BATCH_IMAGES, 100, 5)))
    max_data_shape.append(('gt_masks', (config.TRAIN.BATCH_IMAGES, 100, max([v[0] for v in config.SCALES]), max(v[1] for v in config.SCALES))))
    print 'providing maximum shape', max_data_shape, max_label_shape

    # infer shape
    data_shape_dict = dict(train_data.provide_data_single + train_data.provide_label_single)
    print 'data shape:'
    pprint.pprint(data_shape_dict)
    sym_instance.infer_shape(data_shape_dict)

    # load and initialize params
    if config.TRAIN.RESUME:
        print 'continue training from ', begin_epoch
        arg_params, aux_params = load_param(prefix, begin_epoch, convert=True)
    else:
        arg_params, aux_params = load_param(pretrained, epoch, convert=True)
        sym_instance.init_weight(config, arg_params, aux_params)

    # check parameter shapes
    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict)

    # create solver
    fixed_param_prefix = config.network.FIXED_PARAMS
    data_names = [k[0] for k in train_data.provide_data_single]
    label_names = [k[0] for k in train_data.provide_label_single]

    mod = MutableModule(sym, data_names=data_names, label_names=label_names,
                        logger=logger, context=ctx, max_data_shapes=[max_data_shape for _ in xrange(batch_size)],
                        max_label_shapes=[max_label_shape for _ in xrange(batch_size)], fixed_param_prefix=fixed_param_prefix)

    # decide training metric
    # RPN, classification accuracy/loss, regression loss
    rpn_acc = metric.RPNAccMetric()
    rpn_cls_loss = metric.RPNLogLossMetric()
    rpn_bbox_loss = metric.RPNL1LossMetric()

    fcis_acc = metric.FCISAccMetric(config)
    fcis_acc_fg = metric.FCISAccFGMetric(config)
    fcis_cls_loss = metric.FCISLogLossMetric(config)
    fcis_bbox_loss = metric.FCISL1LossMetric(config)
    fcis_mask_loss = metric.FCISMaskLossMetric(config)

    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [rpn_acc, rpn_cls_loss, rpn_bbox_loss,
                         fcis_acc, fcis_acc_fg, fcis_cls_loss, fcis_bbox_loss, fcis_mask_loss]:
        eval_metrics.add(child_metric)

    batch_end_callback = callback.Speedometer(train_data.batch_size, frequent=args.frequent)
    means = np.tile(np.array(config.TRAIN.BBOX_MEANS), 2 if config.CLASS_AGNOSTIC else config.dataset.NUM_CLASSES)
    stds = np.tile(np.array(config.TRAIN.BBOX_STDS), 2 if config.CLASS_AGNOSTIC else config.dataset.NUM_CLASSES)
    epoch_end_callback = callback.do_checkpoint(prefix, means, stds)

    # print epoch, begin_epoch, end_epoch, lr_step
    base_lr = lr
    lr_factor = 0.1
    lr_epoch = [float(epoch) for epoch in lr_step.split(',')]
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(sdsdb) / batch_size) for epoch in lr_epoch_diff]
    print 'lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters
    lr_scheduler = WarmupMultiFactorScheduler(lr_iters, lr_factor, config.TRAIN.warmup, config.TRAIN.warmup_lr, config.TRAIN.warmup_step)
    # optimizer
    optimizer_params = {'momentum': config.TRAIN.momentum,
                        'wd': config.TRAIN.wd,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': 1.0,
                        'clip_gradient': None}

    if not isinstance(train_data, PrefetchingIter):
        train_data = PrefetchingIter(train_data)

    # del sdsdb
    # a = mx.viz.plot_network(sym)
    # a.render('../example', view=True)
    # print 'prepare sds finished'

    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=config.default.kvstore,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch)


def main():
    print 'Called with argument:', args
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    train_net(args, ctx, config.network.pretrained, config.network.pretrained_epoch,
              config.TRAIN.model_prefix, config.TRAIN.begin_epoch, config.TRAIN.end_epoch,
              config.TRAIN.lr, config.TRAIN.lr_step)

if __name__ == '__main__':
    main()
