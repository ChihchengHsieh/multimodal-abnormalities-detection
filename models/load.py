import os, pickle, torch
import torch.nn as nn
from typing import Dict, Tuple, Union
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

    model = create_model_from_setup(labels_cols, train_info.model_setup, **kwargs)
    model.to(device)

    model.load_state_dict(
        torch.load(
            os.path.join("trained_models", model_select.value), map_location=device
        )
    )

    optim_path = os.path.join(
        os.path.join("trained_models", f"{train_info.final_model_path}_optim")
    )

    if os.path.isfile(optim_path):
        print("Found optimizer for this model.")
        optim: torch.optim.optimizer.Optimizer = get_optimiser(model,train_info.model_setup)
        optim.load_state_dict(
            torch.load(
                os.path.join("trained_models", f"{train_info.final_model_path}_optim"),
                map_location=device,
            )
        )
    else:
        optim = None
        print("No optimizer found for this model.")

    return model, train_info, optim

def get_current_epoch(trained_model: TrainedModels) -> int:
    return int(
        (
            [substr for substr in trained_model.value.split("_") if "epoch" in substr][
                0
            ]
        ).replace("epoch", "")
    )

def get_model_name(trained_model: TrainedModels, naming_map: Dict[TrainedModels, str]= None) -> str:
   return naming_map[trained_model] if naming_map else str(trained_model).split(".")[-1]



def get_model_label(trained_modelL: TrainedModels, naming_map: Dict[TrainedModels, str])-> str:
    return get_model_name(trained_modelL, naming_map) + f" (epoch: {get_current_epoch(trained_modelL)})" 

def get_dataset_label(dataset, select_model):
    return dataset + f" (epoch: {get_current_epoch(select_model)})"