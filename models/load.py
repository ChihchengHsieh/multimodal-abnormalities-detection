import os, pickle, torch
import torch.nn as nn
from typing import Dict, Tuple, Union
from models.dynamic_loss import DynamicWeightedLoss
from utils.train import get_optimiser

from .build import create_model_from_setup
from .train import TrainedModels, TrainingInfo
from torch.optim.optimizer import Optimizer
from .setup import ModelSetup


def get_trained_model(
    model_select: TrainedModels, labels_cols, device, **kwargs,
) -> Tuple[nn.Module, TrainingInfo, Union[Optimizer, None]]:

    with open(os.path.join("training_records", f"{model_select.value}.pkl"), "rb") as f:
        train_info: TrainingInfo = pickle.load(f)

    model = create_model_from_setup(labels_cols, train_info.model_setup, **kwargs,)
    model.to(device)

    cp: Dict = torch.load(
        os.path.join("trained_models", model_select.value), map_location=device
    )

    model.load_state_dict(cp["model_state_dict"])

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    dynamic_loss_weight = None
    if "dynamic_weight_state_dict" in cp:
        loss_keys = [
            "loss_classifier",
            "loss_box_reg",
            "loss_objectness",
            "loss_rpn_box_reg",
        ]

        dynamic_loss_weight = DynamicWeightedLoss(
            keys=loss_keys + ["loss_mask"]
            if train_info.model_setup.use_mask
            else loss_keys
        )
        dynamic_loss_weight.to(device)
        dynamic_loss_weight.load_state_dict(cp["dynamic_weight_state_dict"])
        params += [p for p in dynamic_loss_weight.parameters() if p.requires_grad]

    optim = None
    if "optimizer_state_dict" in cp:
        optim: torch.optim.optimizer.Optimizer = get_optimiser(
            params, train_info.model_setup
        )
        optim.load_state_dict(cp["optimizer_state_dict"])

    return model, train_info, optim, dynamic_loss_weight
    # return model, train_info, None, None


def get_current_epoch(trained_model: TrainedModels) -> int:
    return int(
        (
            [substr for substr in trained_model.value.split("_") if "epoch" in substr][
                0
            ]
        ).replace("epoch", "")
    )


def get_model_name(
    trained_model: TrainedModels, naming_map: Dict[TrainedModels, str] = None
) -> str:
    return (
        naming_map[trained_model] if naming_map else str(trained_model).split(".")[-1]
    )


def get_model_label(
    trained_modelL: TrainedModels, naming_map: Dict[TrainedModels, str]
) -> str:
    return (
        get_model_name(trained_modelL, naming_map)
        + f" (epoch: {get_current_epoch(trained_modelL)})"
    )


def get_dataset_label(dataset, select_model):
    return dataset + f" (epoch: {get_current_epoch(select_model)})"
