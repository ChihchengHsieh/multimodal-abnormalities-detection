import PIL
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from pandas import test
import torch.nn as nn

from typing import Callable, Dict, List, Union, Tuple
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib import colors
from utils.pred import pred_thrs_check
from utils.save import get_data_from_metric_logger
from data.datasets import ReflacxDataset, collate_fn
from utils.detect_utils import MetricLogger
from utils.coco_eval import CocoEvaluator, external_summarize


def transparent_cmap(
    cmap: colors.LinearSegmentedColormap, N: int = 255
) -> colors.LinearSegmentedColormap:

    "Copy colormap and set alpha values"

    t_cmap = cmap
    t_cmap._init()
    t_cmap._lut[:, -1] = np.linspace(0, 0.8, N + 4)

    return t_cmap


DISEASE_CMAP: Dict = {
    "transparent": {
        "Enlarged cardiac silhouette": transparent_cmap(plt.cm.autumn),
        "Atelectasis": transparent_cmap(plt.cm.Reds),
        "Pleural abnormality": transparent_cmap(plt.cm.Oranges),
        "Consolidation": transparent_cmap(plt.cm.Greens),
        "Pulmonary edema": transparent_cmap(plt.cm.Blues),
    },
    "solid": {
        "Enlarged cardiac silhouette": "yellow",
        "Atelectasis": "red",
        "Pleural abnormality": "orange",
        "Consolidation": "lightgreen",
        "Pulmonary edema": "dodgerblue",
    },
}


def get_legend_elements(disease_cmap_solid: Dict[str, str]) -> List[Line2D]:
    legend_elements = []
    for k, v in disease_cmap_solid.items():
        legend_elements.append(Line2D([0], [0], color=v, lw=4, label=k))

    return legend_elements


def plot_losses(
    train_logers: Union[List[MetricLogger], List[Dict]],
    val_logers: Union[List[MetricLogger], List[Dict]],
    test_logers: Union[List[MetricLogger], List[Dict], None] = None,
):
    if isinstance(train_logers[0], MetricLogger):
        train_data = [get_data_from_metric_logger(loger) for loger in train_logers]
    else:
        train_data = train_logers

    if isinstance(val_logers[0], MetricLogger):
        val_data = [get_data_from_metric_logger(loger) for loger in val_logers]
    else:
        val_data = val_logers

    if test_logers and isinstance(test_logers[0], MetricLogger):
        test_data = [get_data_from_metric_logger(loger) for loger in test_logers]
    else:
        test_data = test_logers


    train_data_keys = train_data[0].keys()

    fig, subplots = plt.subplots(
        len(train_data_keys),
        figsize=(10, 5 * len(train_data_keys)),
        dpi=80,
        sharex=True,
    )

    fig.suptitle(f"Training Losses")

    for i, k in enumerate(train_data_keys):
        subplots[i].set_title(k)
        subplots[i].plot(
            [data[k] for data in train_data],
            marker="o",
            label="train",
            color="steelblue",
        )
        
        if k in val_data[0].keys():
            subplots[i].plot(
                [data[k] for data in val_data], marker="o", label="val", color="orange"
            )

        if k in test_data[0].keys():
            subplots[i].plot(
                [data[k] for data in test_data], marker="o", label="test", color="red"
            )

        subplots[i].legend(loc="upper left")

    subplots[-1].set_xlabel("Epoch")
    plt.plot()
    plt.pause(0.01)


def plot_loss(train_logers: Union[List[MetricLogger], List[Dict]]):
    if isinstance(train_logers[0], MetricLogger):
        train_data = [get_data_from_metric_logger(loger) for loger in train_logers]
    else:
        train_data = train_logers

    train_data_keys = train_data[0].keys()

    fig, subplots = plt.subplots(
        len(train_data_keys),
        figsize=(10, 5 * len(train_data_keys)),
        dpi=80,
        sharex=True,
    )

    fig.suptitle(f"Training Losses")

    for i, k in enumerate(train_data_keys):
        subplots[i].set_title(k)
        subplots[i].plot(
            [data[k] for data in train_data], marker="o", label=k, color="steelblue"
        )
        # subplots[i].legend(loc="upper left")

    subplots[-1].set_xlabel("Epoch")
    plt.plot()
    plt.pause(0.01)


def plot_evaluator(
    evaluators: List[CocoEvaluator],
    iouThr: float = 0.5,
    areaRng: str = "all",
    maxDets: int = 10,
) -> Figure:

    all_precisions: List[float] = []
    all_recalls: List[float] = []

    for i in range(len(evaluators)):

        all_precisions.append(
            external_summarize(
                evaluators[i].coco_eval["bbox"],
                ap=1,
                iouThr=iouThr,
                areaRng=areaRng,
                maxDets=maxDets,
                print_result=False,
            )
        )

        all_recalls.append(
            external_summarize(
                evaluators[i].coco_eval["bbox"],
                ap=0,
                iouThr=iouThr,
                areaRng=areaRng,
                maxDets=maxDets,
                print_result=False,
            )
        )

    fig, (precision_ax, recall_ax) = plt.subplots(
        2, figsize=(10, 10), dpi=80, sharex=True,
    )

    precision_ax.set_title("Precision")
    precision_ax.plot(
        all_precisions, marker="o", label="Precision", color="darkorange",
    )
    precision_ax.legend(loc="upper left")
    recall_ax.set_title("Recall")
    recall_ax.plot(
        all_recalls, marker="o", label="Recall", color="darkorange",
    )
    recall_ax.legend(loc="upper left")

    recall_ax.set_xlabel("Epoch")

    plt.plot()
    plt.pause(0.01)

    return fig


def plot_ap_ars(
    train_ap_ars: List[Dict[str, float]],
    val_ap_ars: List[Dict[str, float]],
    test_ap_ars=None,
    fig_title=None,
) -> Figure:
    """
    Plot both training and validation evaluator during training to check overfitting.
    """

    fig, (precision_ax, recall_ax) = plt.subplots(
        2, figsize=(10, 10), dpi=80, sharex=True,
    )

    if fig_title:
        fig.suptitle(f"{fig_title}")

    precision_ax.set_title("Precision")
    precision_ax.plot(
        [ap_ar["ap"] for ap_ar in train_ap_ars],
        marker="o",
        label="train",
        color="royalblue",
    )
    precision_ax.plot(
        [ap_ar["ap"] for ap_ar in val_ap_ars],
        marker="o",
        label="validation",
        color="darkorange",
    )
    if test_ap_ars:
        precision_ax.plot(
            [ap_ar["ap"] for ap_ar in test_ap_ars],
            marker="o",
            label="test",
            color="red",
        )

    precision_ax.legend(loc="upper left")

    recall_ax.set_title("Recall")
    recall_ax.plot(
        [ap_ar["ar"] for ap_ar in train_ap_ars],
        marker="o",
        label="train",
        color="royalblue",
    )

    recall_ax.plot(
        [ap_ar["ar"] for ap_ar in val_ap_ars],
        marker="o",
        label="validation",
        color="darkorange",
    )

    if test_ap_ars:
        recall_ax.plot(
            [ap_ar["ar"] for ap_ar in test_ap_ars],
            marker="o",
            label="test",
            color="red",
        )

    recall_ax.legend(loc="upper left")

    recall_ax.set_xlabel("Epoch")

    plt.plot()
    plt.pause(0.01)

    return fig


def plot_train_val_evaluators(
    train_evaluators: List[CocoEvaluator],
    val_evaluators: List[CocoEvaluator],
    iouThr=0.5,
    areaRng="all",
    maxDets=10,
) -> Figure:
    """
    Plot both training and validation evaluator during training to check overfitting.
    """

    train_precisions: List[float] = []
    train_recalls: List[float] = []

    val_precisions: List[float] = []
    val_recalls: List[float] = []

    for i in range(len(train_evaluators)):
        train_precisions.append(
            external_summarize(
                train_evaluators[i].coco_eval["bbox"],
                ap=1,
                iouThr=iouThr,
                areaRng=areaRng,
                maxDets=maxDets,
                print_result=False,
            )
        )

        val_precisions.append(
            external_summarize(
                val_evaluators[i].coco_eval["bbox"],
                ap=1,
                iouThr=iouThr,
                areaRng=areaRng,
                maxDets=maxDets,
                print_result=False,
            )
        )

        train_recalls.append(
            external_summarize(
                train_evaluators[i].coco_eval["bbox"],
                ap=0,
                iouThr=iouThr,
                areaRng=areaRng,
                maxDets=maxDets,
                print_result=False,
            )
        )

        val_recalls.append(
            external_summarize(
                val_evaluators[i].coco_eval["bbox"],
                ap=0,
                iouThr=iouThr,
                areaRng=areaRng,
                maxDets=maxDets,
                print_result=False,
            )
        )

    fig, (precision_ax, recall_ax) = plt.subplots(
        2, figsize=(10, 10), dpi=80, sharex=True,
    )

    precision_ax.set_title("Precision")
    precision_ax.plot(
        train_precisions, marker="o", label="train", color="royalblue",
    )
    precision_ax.plot(
        val_precisions, marker="o", label="validation", color="darkorange",
    )
    precision_ax.legend(loc="upper left")

    recall_ax.set_title("Recall")
    recall_ax.plot(
        train_recalls, marker="o", label="train", color="royalblue",
    )

    recall_ax.plot(
        val_recalls, marker="o", label="validation", color="darkorange",
    )
    recall_ax.legend(loc="upper left")

    recall_ax.set_xlabel("Epoch")

    plt.plot()
    plt.pause(0.01)

    return fig


def plot_seg(
    target: List[Dict],
    pred: List[Dict],
    label_idx_to_disease: Callable[[int], str],
    legend_elements: List[Line2D],
    transparent_disease_color_code_map: Dict[str, colors.LinearSegmentedColormap],
    seg_thres: float = 0,
) -> Figure:
    """
    Plot segmentation prediction.
    """

    fig, (gt_ax, pred_ax) = plt.subplots(1, 2, figsize=(20, 10), dpi=80, sharex=True)

    fig.suptitle(target["image_path"])

    img = PIL.Image.open(target["image_path"]).convert("RGB")

    gt_ax.imshow(img)
    gt_ax.set_title("Ground Truth")
    pred_ax.imshow(img)
    pred_ax.set_title("Predictions")

    fig.legend(handles=legend_elements, loc="upper right")

    for label, m in zip(
        target["labels"].detach().cpu().numpy(), target["masks"].detach().cpu().numpy(),
    ):
        disease = label_idx_to_disease(label)
        mask_img = PIL.Image.fromarray(m * 255)
        gt_ax.imshow(
            mask_img,
            transparent_disease_color_code_map[disease],
            interpolation="none",
            alpha=0.7,
        )

    for label, m in zip(
        pred[0]["labels"].detach().cpu().numpy(),
        pred[0]["masks"].detach().cpu().numpy(),
    ):
        disease = label_idx_to_disease(label)
        mask = (m.squeeze() > seg_thres).astype(np.uint8)
        mask_img = PIL.Image.fromarray(mask * 255)

        pred_ax.imshow(
            mask_img,
            transparent_disease_color_code_map[disease],
            interpolation="none",
            alpha=0.7,
        )

    return fig


def plot_bbox(
    target: List[Dict],
    pred: List[Dict],
    label_idx_to_disease: Callable[[int], str],
    legend_elements: List[Line2D],
    disease_color_code_map: Dict[str, str],
) -> Figure:

    fig, (gt_ax, pred_ax) = plt.subplots(1, 2, figsize=(20, 10), dpi=80, sharex=True)

    fig.suptitle(target["image_path"])

    fig.legend(handles=legend_elements, loc="upper right")

    img = PIL.Image.open(target["image_path"]).convert("RGB")

    gt_ax.imshow(img)
    gt_ax.set_title(f"Ground Truth ({len(target['boxes'].detach().cpu().numpy())})")
    pred_ax.imshow(img)
    pred_ax.set_title(f"Predictions ({len(pred['boxes'].detach().cpu().numpy())})")

    # load image
    gt_recs = []
    pred_recs = []

    for label, bbox, score in zip(
        pred["labels"].detach().cpu().numpy(),
        pred["boxes"].detach().cpu().numpy(),
        pred["scores"].detach().cpu().numpy(),
    ):
        disease = label_idx_to_disease(label)
        c = disease_color_code_map[disease]
        pred_recs.append(
            Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                fill=False,
                color=c,
                linewidth=2,
            )
        )
        pred_ax.text(
            bbox[0],
            bbox[1],
            f"{disease} ({score:.2f})",
            color="black",
            backgroundcolor=c,
        )

    for rec in pred_recs:
        pred_ax.add_patch(rec)

    for label, bbox in zip(
        target["labels"].detach().cpu().numpy(), target["boxes"].detach().cpu().numpy()
    ):
        disease = label_idx_to_disease(label)
        c = disease_color_code_map[disease]
        gt_recs.append(
            Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                fill=False,
                color=c,
                linewidth=2,
            )
        )
        gt_ax.text(bbox[0], bbox[1], disease, color="black", backgroundcolor=c)

    for rec in gt_recs:
        gt_ax.add_patch(rec)

    plt.plot()
    plt.pause(0.01)

    return fig


def plot_result(
    model: nn.Module,
    dataset: ReflacxDataset,
    device: str,
    idx: int,
    legend_elements: List[Line2D],
    disease_cmap=DISEASE_CMAP,
    seg=False,
    seg_thres=0.5,
    score_thres: Dict = None,
) -> Tuple[Figure, Union[Figure, None]]:
    model.eval()
    data = collate_fn([dataset[idx]])
    data = dataset.prepare_input_from_data(data, device)
    target = data[-1]
    pred = model(*data[:-1])
    pred = pred[0]

    if not score_thres is None:
        pred = pred_thrs_check(pred, dataset, score_thres, device)

    bb_fig = plot_bbox(
        target[0],
        pred,
        dataset.label_idx_to_disease,
        legend_elements,
        disease_cmap["solid"],
    )

    seg_fig = None

    if seg:
        seg_fig = plot_seg(
            target[0],
            pred,
            dataset.label_idx_to_disease,
            legend_elements,
            disease_cmap["transparent"],
            seg_thres=seg_thres,
        )

    return bb_fig, seg_fig

