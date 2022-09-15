import math, sys, time, torch, torchvision
from typing import Dict, List, Tuple
import torch.nn as nn

from models.setup import ModelSetup

from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator

from . import detect_utils
from data.helpers import map_target_to_device

from models.detectors.rcnn import MultimodalMaskRCNN
from .pred import pred_thrs_check
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer

cpu_device = torch.device("cpu")


def get_iou_types(model: nn.Module, setup: ModelSetup) -> List[str]:
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if (
        isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN)
        or isinstance(model_without_ddp, MultimodalMaskRCNN)
    ) and setup.use_mask:
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def loss_multiplier( loss_dict, epoch = None):
    # in the first 100 epochs, we can train the rpn first.
    # if epoch and epoch < 10:
    #     loss_dict["loss_classifier"] = loss_dict["loss_classifier"].detach()
    #     loss_dict["loss_box_reg"]  = loss_dict["loss_box_reg"].detach()
    #     loss_dict["loss_objectness"] *= 1
    #     loss_dict["loss_rpn_box_reg"] *= 1
    # else:
    #     loss_dict["loss_classifier"] *= 1
    #     loss_dict["loss_box_reg"] *= 1
    #     loss_dict["loss_objectness"] *= 1
    #     loss_dict["loss_rpn_box_reg"] *= 1
    # loss_dict["loss_objectness"] = loss_dict["loss_objectness"].detach()
    # loss_dict["loss_rpn_box_reg"]  = loss_dict["loss_rpn_box_reg"].detach()

    return loss_dict


def xami_train_one_epoch(
    setup: ModelSetup,
    model: nn.Module,
    optimizer: Optimizer,
    data_loader: DataLoader,
    device: str,
    epoch: int,
    print_freq: int,
    iou_types: List[str],
    coco: Dataset,
    score_thres: Dict[str, float] = None,
    evaluate_on_run=True,
    params_dict: Dict = None,
    dynamic_loss_weight=None,
) -> Tuple[CocoEvaluator, detect_utils.MetricLogger]:
    model.train()
    metric_logger = detect_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", detect_utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = f"Epoch: [{epoch}]"

    if evaluate_on_run:
        coco_evaluator = CocoEvaluator(coco, iou_types, params_dict)

    lr_scheduler = None

    # if epoch == 1:
    #     print("start wariming up ")
    #     warmup_factor = 1.0 / 1000
    #     warmup_iters = min(1000, len(data_loader) - 1)

    #     lr_scheduler = torch.optim.lr_scheduler.LinearLR(
    #         optimizer, start_factor=warmup_factor, total_iters=warmup_iters
    #     )

    for data in metric_logger.log_every(data_loader, print_freq, header):
        data = data_loader.dataset.prepare_input_from_data(data, device)
        with torch.cuda.amp.autocast(enabled=False):
            loss_dict, outputs = model(*data[:-1], targets=data[-1])
            loss_dict = loss_multiplier(loss_dict,epoch)

            if dynamic_loss_weight:
                # loss_dict["loss_objectness"] *= 4
                # loss_dict["loss_rpn_box_reg"] *= 2
                losses = dynamic_loss_weight(loss_dict)
            else:
                losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = detect_utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()

        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if evaluate_on_run:

            if not score_thres is None:
                outputs = [
                    pred_thrs_check(pred, data_loader.dataset, score_thres, device)
                    for pred in outputs
                ]

            outputs = [
                {k: v.detach().to(cpu_device) for k, v in t.items()} for t in outputs
            ]

            res = {
                target["image_id"].item(): output
                for target, output in zip(data[-1], outputs)
            }
            coco_evaluator.update(res)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if evaluate_on_run:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        return coco_evaluator, metric_logger

    return metric_logger


@torch.inference_mode()
def xami_evaluate(
    setup: ModelSetup, 
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    coco: Dataset,
    iou_types: List[str],
    params_dict: Dict = None,
    score_thres: Dict[str, float] = None,
) -> Tuple[CocoEvaluator, detect_utils.MetricLogger]:

    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)

    model.eval()
    metric_logger = detect_utils.MetricLogger(delimiter="  ")
    header = "Evaluation:"
    coco_evaluator = CocoEvaluator(coco, iou_types, params_dict)

    for data in metric_logger.log_every(data_loader, 100, header):
        data = data_loader.dataset.prepare_input_from_data(data, device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        loss_dict, outputs = model(*data[:-1], targets=data[-1])
        loss_dict = loss_multiplier(loss_dict)

        loss_dict_reduced = detect_utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        if not score_thres is None:
            outputs = [
                pred_thrs_check(pred, data_loader.dataset, score_thres, device)
                for pred in outputs
            ]

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {
            target["image_id"].item(): output
            for target, output in zip(data[-1], outputs)
        }

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator, metric_logger

