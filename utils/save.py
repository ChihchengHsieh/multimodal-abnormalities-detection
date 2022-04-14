import os
from typing import Dict, Tuple
import torch
import pickle

from datetime import datetime
from .eval import get_ar_ap
from copy import deepcopy
from utils.detect_utils import MetricLogger
from utils.engine import xami_evaluate
import torch.nn as nn

import utils.print as print_f
from models.load import TrainingInfo
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


def get_train_data(loger: MetricLogger) -> Dict[str, float]:
    train_data = {}
    for k in loger.meters.keys():
        train_data[k] = loger.meters[k].avg

    return train_data

###########################################################
def save_checkpoint(
    train_info: TrainingInfo,
    model: nn.Module,
    val_ar: float,
    val_ap: float,
    test_ar: float,
    test_ap: float,
    optim: Optimizer = None,
) -> TrainingInfo:
    current_time_string = datetime.now().strftime("%m-%d-%Y %H-%M-%S")

    model_path = (
        (
            f"val_ar_{val_ar:.4f}_ap_{val_ap:.4f}_"
            + f"test_ar_{test_ar:.4f}_ap_{test_ap:.4f}_"
            + f"epoch{train_info.epoch}_{train_info.clinical_cond}Clincal_{current_time_string}"
            + f"_{train_info.model_setup.name}"
        )
        .replace(":", "_")
        .replace(".", "_")
    )

    train_info.final_model_path = model_path

    torch.save(
        model.state_dict(),
        os.path.join(os.path.join("trained_models", train_info.final_model_path)),
    )

    # Save optimizer if necessary.
    if optim:
        torch.save(
            optim.state_dict(),
            os.path.join(
                os.path.join("trained_models", f"{train_info.final_model_path}_optim")
            ),
        )

    with open(
        os.path.join("training_records", f"{train_info.final_model_path }.pkl"), "wb",
    ) as train_info_f:
        pickle.dump(train_info, train_info_f)

    return train_info


def remove_previous_model(previous_model: str):
    if not previous_model is None:
        # delete previous model
        if os.path.exists(os.path.join(os.path.join("trained_models", previous_model))):
            os.remove(os.path.join(os.path.join("trained_models", previous_model)))
        # delete previous training records.
        if os.path.exists(os.path.join("training_records", f"{previous_model}.pkl")):
            os.remove(os.path.join("training_records", f"{previous_model}.pkl"))
        print(f"Previous model: [{previous_model}] has been remove!!")


def check_best(
    train_info: TrainingInfo,
    eval_params_dict: Dict,
    model: nn.Module,
    optim: Optimizer,
    test_dataloader: DataLoader,
    device: str,
) -> Tuple[float, float, TrainingInfo]:
    val_ar, val_ap = get_ar_ap(train_info.val_evaluators[-1])

    ## Targeting the model with higher Average Recall and Average Precision.
    if val_ar > train_info.best_val_ar or val_ap > train_info.best_val_ap:

        train_info.test_evaluator = xami_evaluate(
            model, test_dataloader, device=device, params_dict=eval_params_dict
        )

        test_ar, test_ap = get_ar_ap(train_info.test_evaluator)

        if val_ar > train_info.best_val_ar:
            ## Save best validation model
            previous_ar_model = deepcopy(train_info.best_ar_val_model_path)
            train_info = save_checkpoint(
                train_info=train_info,
                model=model,
                val_ar=val_ar,
                val_ap=val_ap,
                test_ar=test_ar,
                test_ap=test_ap,
                optim=optim,
            )
            train_info.best_ar_val_model_path = train_info.final_model_path
            train_info.best_val_ar = val_ar
            remove_previous_model(previous_ar_model)

        if val_ap > train_info.best_val_ap:
            previous_ap_model = deepcopy(train_info.best_ap_val_model_path)
            train_info = save_checkpoint(
                train_info=train_info,
                model=model,
                val_ar=val_ar,
                val_ap=val_ap,
                test_ar=test_ar,
                test_ap=test_ap,
                optim=optim,
            )
            train_info.best_ap_val_model_path = train_info.final_model_path
            train_info.best_val_ap = val_ap
            remove_previous_model(previous_ap_model)

    return val_ar, val_ap, train_info


def end_train(
    train_info: TrainingInfo,
    model: nn.Module,
    optim: Optimizer,
    eval_params_dict: Dict,
    last_val_ar: float,
    last_val_ap: float,
    test_dataloader: DataLoader,
    device: str,
) -> TrainingInfo:
    train_info.end_t = datetime.now()
    sec_took = (train_info.end_t - train_info.start_t).seconds

    print_f.print_title(
        f"| Training Done, start testing! | [{train_info.epoch}] Epochs Training time: [{sec_took}] seconds, Avg time / Epoch: [{sec_took/train_info.epoch}] seconds"
    )

    # print model
    if train_info.model_setup.use_early_stop_model:
        print_f.print_title(
            f"Best AP validation model has been saved to: [{train_info.best_ap_val_model_path}]"
        )
        print_f.print_title(
            f"Best AR validation model has been saved to: [{train_info.best_ar_val_model_path}]"
        )

    train_info.test_evaluator = xami_evaluate(
        model, test_dataloader, device=device, params_dict=eval_params_dict
    )

    test_ar, test_ap = get_ar_ap(train_info.test_evaluator)

    train_info = save_checkpoint(
        train_info=train_info,
        model=model,
        val_ar=last_val_ar,
        val_ap=last_val_ap,
        test_ar=test_ar,
        test_ap=test_ap,
        optim=optim,
    )

    print_f.print_title(
        f"The final model has been saved to: [{train_info.final_model_path}]"
    )
    return train_info

