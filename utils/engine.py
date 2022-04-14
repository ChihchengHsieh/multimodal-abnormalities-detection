import math, sys, time, torch, torchvision
from typing import Dict, List
import torch.nn as nn

from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator

from . import detect_utils
from .helpers import map_target_to_device

from models.detectors.rcnn import MultimodalMaskRCNN
from .pred import pred_thrs_check
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer



def _get_iou_types(model: nn.Module) -> List[str]:
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(
        model_without_ddp, torchvision.models.detection.MaskRCNN
    ) or isinstance(model_without_ddp, MultimodalMaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def xami_train_one_epoch(
    model: nn.Module,
    optimizer: Optimizer,
    data_loader: DataLoader,
    device: str,
    epoch: int,
    print_freq: int,
) -> detect_utils.MetricLogger:
    model.train()
    metric_logger = detect_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", detect_utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for data in metric_logger.log_every(data_loader, print_freq, header):
        data = data_loader.dataset.prepare_input_from_data(data, device)
        with torch.cuda.amp.autocast(enabled=False):
            loss_dict = model(*data[:-1], targets=data[-1])
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

    return metric_logger


@torch.inference_mode()
def xami_evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    params_dict: Dict = None,
    score_thres: Dict[str, float] = None,
) -> CocoEvaluator:
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = detect_utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types, params_dict)

    for data in metric_logger.log_every(data_loader, 100, header):
        data = data_loader.dataset.prepare_input_from_data(data, device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(*data[:-1])

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
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

