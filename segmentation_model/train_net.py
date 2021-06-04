import os
import time
import torch
import datetime
import logging
import numpy as np
import copy
from torch import nn


import detectron2.engine as engine
from detectron2.utils.logger import log_every_n_seconds
import detectron2.utils.comm as comm
from detectron2.data import DatasetMapper,build_detection_test_loader, build_detection_train_loader
from detectron2.layers import paste_masks_in_image
from detectron2.structures import Instances
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.structures import ImageList, Instances
from typing import Tuple, Dict
from detectron2.modeling.backbone.build import build_backbone
import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
from detectron2.engine import hooks as hk
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluators
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.utils.comm import get_world_size, is_main_process
from contextlib import ExitStack, contextmanager

from config import config
from PIL import Image

BYTES_PER_FLOAT = 4

# determine it based on available resources.
GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

class Basehook(engine.HookBase):
    def __init__(self, eval_period, model, data_loader,evaluator):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self.evaluator = evaluator

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        # results = inference_on_dataset(self.model, self._data_loader, self.evaluator )

        num_devices = get_world_size()
        logger = logging.getLogger(__name__)
        logger.info("Start inference on {} images".format(len(self._data_loader)))

        total = len(self._data_loader)  # inference data loader must have a fixed length
        # if self.evaluator is None:
        #     # create a no-op evaluator
        #     self.evaluator = DatasetEvaluators([])
        # self.evaluator.reset()

        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_compute_time = 0
        # with ExitStack() as stack:
        #     if isinstance(self._model, nn.Module):
        #         stack.enter_context(inference_context(self._model))
        #     stack.enter_context(torch.no_grad())

        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = self._model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            self.evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

            #Add custom loss
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)

        # Measure the time only for this worker (before the synchronization barrier)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        # NOTE this format is parsed by grep
        logger.info(
            "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
                total_time_str, total_time / (total - num_warmup), num_devices
            )
        )
        total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
        logger.info(
            "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
                total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
            )
        )

        # results = self.evaluator.evaluate()
        # # An evaluator may return None when not in main process.
        # # Replace it by an empty dict instead to make it easier for downstream code to handle
        # if results is None:
        #     results = {}
        #
        # # copy from hooks.py
        #
        # assert isinstance(
        #     results, dict
        # ), "Eval function must return a dict. Got {} instead.".format(results)
        #
        # flattened_results = flatten_results_dict(results)
        # for k, v in flattened_results.items():
        #     try:
        #         v = float(v)
        #     except Exception as e:
        #         raise ValueError(
        #             "[EvalHook] eval_function should return a nested dict of float. "
        #             "Got '{}: {}' instead.".format(k, v)
        #         ) from e
        # self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)

        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.

        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        """
        Called after each iteration.
        """

        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)

    def before_train(self):
        """
        Called before the first iteration.
        """

        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        # print(f"Hello at iteration {self.trainer.iter}!")



def detector_postprocess(
    results: Instances, output_height: int, output_width: int, mask_threshold: float = 0.5
):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    # Change to 'if is_tracing' after PT1.7
    if isinstance(output_height, torch.Tensor):
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances(new_size, **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        results.pred_masks = retry_if_cuda_oom(paste_masks_in_image)(
            results.pred_masks[:, 0, :, :],  # N, 1, M, M
            results.pred_boxes,
            results.image_size,
            threshold=mask_threshold,
        )

    return results


def _postprocess(instances, batched_inputs: Tuple[Dict[str, torch.Tensor]], image_sizes, mask_threshold: float = 0.5):
    """
    Rescale the output instances to the target size.
    """
    # note: private function; subject to changes
    processed_results = []
    for results_per_image, input_per_image, image_size in zip(
        instances, batched_inputs, image_sizes
    ):
        height = input_per_image.get("height", image_size[0])
        width = input_per_image.get("width", image_size[1])
        r = detector_postprocess(results_per_image, height, width, mask_threshold)
        processed_results.append({"instances": r})
    return processed_results

def preprocess_image(batched_inputs: Tuple[Dict[str, torch.Tensor]],cfg):
    """
    Normalize, pad and batch the input images.
    """
    pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
    pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
    backbone = build_backbone(cfg)

    images = [x["image"].to(pixel_mean.device) for x in batched_inputs]
    images = [(x - pixel_mean) / pixel_std for x in images]
    images = ImageList.from_tensors(images, backbone.size_divisibility)
    return images

class BasePredictor(engine.DefaultPredictor):
    """
        Class overload from DefaultPredictor for more custom functionalities
    """

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model.inference([inputs],do_postprocess=False)
            images = preprocess_image([inputs],self.cfg)
            predictions = _postprocess(predictions, [inputs], images.image_sizes,mask_threshold = 0.5)

            return predictions[0]


def custom_mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    augs = T.AugmentationList([
        T.Resize((224,224),interp=Image.LANCZOS),
        # T.RandomBrightness(0.5, 2),
        # T.RandomContrast(0.5, 2),
        # T.RandomSaturation(0.5, 2),
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False)
    ])

    auginput = T.AugInput(image)
    transform = augs(auginput)
    image = torch.as_tensor(auginput.image.transpose(2, 0, 1).astype("float32"))
    dataset_dict["image"] = image

    annos = [
        utils.transform_instance_annotations(obj, [transform], image.shape[1:])
        for obj in dataset_dict.pop("annotations")
    ]
    instances = utils.annotations_to_instances(annos, image.shape[1:])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


class BaseTrainer(engine.DefaultTrainer):


    def __init__(self, cfg, local_config):
        self.local_config = local_config
        self.validation_period = local_config["eval_period"]
        super().__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

    @classmethod
    def build_evaluator(self, cfg, dataset_name, output_folder=None):
        if config.train_config["validation"]:
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name,mapper=custom_mapper)

    def build_hooks(self):
        hooks = super().build_hooks()
        # for idx, h in enumerate(hooks):
        #     if isinstance(h, hk.EvalHook):
        #         k = idx
        # if(k):
        #     del hooks[k]
        if self.local_config["validation"]:
            hooks.insert(-1, Basehook(
                self.validation_period,
                self.model,
                self.build_test_loader(
                    self.cfg,
                    self.cfg.DATASETS.TEST[0]
                ),
                self.build_evaluator(self.cfg, self.cfg.DATASETS.TEST[0])
            ))
            return hooks

        return hooks
