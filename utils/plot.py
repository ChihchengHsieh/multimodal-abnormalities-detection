import PIL

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from IPython.display import clear_output
from utils.save import get_train_data
from utils.map import map_target_to_device
from data.dataset import collate_fn

from utils.detect_utils import MetricLogger
from utils.coco_eval import external_summarize


def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    t_cmap = cmap
    t_cmap._init()
    t_cmap._lut[:, -1] = np.linspace(0, 0.8, N + 4)

    return t_cmap


disease_cmap = {
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


def get_legend_elements(disease_cmap_solid):
    legend_elements = []
    for k, v in disease_cmap_solid.items():
        legend_elements.append(Line2D([0], [0], color=v, lw=4, label=k))

    return legend_elements


def plot_loss(train_logers):

    clear_output()

    if isinstance(train_logers[0], MetricLogger):
        train_data = [get_train_data(loger) for loger in train_logers]
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
    evaluators, iouThr=0.5, areaRng="all", maxDets=10,
):

    clear_output()

    all_precisions = []
    all_recalls = []

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
        all_precisions,
        marker="o",
        label="Precision",
        color="darkorange",
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


def plot_seg(
    target,
    pred,
    label_idx_to_disease,
    legend_elements,
    transparent_disease_color_code_map,
    seg_thres=0,
):
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


def plot_bbox(
    target, pred, label_idx_to_disease, legend_elements, disease_color_code_map
):

    fig, (gt_ax, pred_ax) = plt.subplots(1, 2, figsize=(20, 10), dpi=80, sharex=True)

    fig.suptitle(target["image_path"])

    fig.legend(handles=legend_elements, loc="upper right")

    img = PIL.Image.open(target["image_path"]).convert("RGB")

    gt_ax.imshow(img)
    gt_ax.set_title(f"Ground Truth ({len(target['boxes'].detach().cpu().numpy())})")
    pred_ax.imshow(img)
    pred_ax.set_title(f"Predictions ({len(pred[0]['boxes'].detach().cpu().numpy())})")

    # load image
    gt_recs = []
    pred_recs = []

    for label, bbox, score in zip(
        pred[0]["labels"].detach().cpu().numpy(),
        pred[0]["boxes"].detach().cpu().numpy(),
        pred[0]["scores"].detach().cpu().numpy(),
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


def plot_result(
    model,
    dataset,
    device,
    idx,
    legend_elements,
    disease_cmap,
    seg=False,
    seg_thres=0.5,
):
    model.eval()
    data = collate_fn([dataset[idx]])
    data = dataset.prepare_input_from_data(data, device)
    target = data[-1]
    pred = model(*data[:-1])

    plot_bbox(
        target[0],
        pred,
        dataset.label_idx_to_disease,
        legend_elements,
        disease_cmap["solid"],
    )

    if seg:
        plot_seg(
            target[0],
            pred,
            dataset.label_idx_to_disease,
            legend_elements,
            disease_cmap["transparent"],
            seg_thres=seg_thres,
        )

