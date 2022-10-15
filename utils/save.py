import os
from typing import Dict, List, Tuple
import torch
import pickle

from datetime import datetime
from copy import deepcopy
from utils.detect_utils import MetricLogger
from utils.engine import xami_evaluate
from utils.eval import get_ap_ar
import torch.nn as nn

import utils.print as print_f
from models.load import TrainingInfo
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset


def get_data_from_metric_logger(loger: MetricLogger) -> Dict[str, float]:
    train_data = {}
    for k in loger.meters.keys():
        train_data[k] = loger.meters[k].avg

    return train_data


###########################################################
# def save_checkpoint(
#     train_info: TrainingInfo,
#     model: nn.Module,
#     val_ar: float,
#     val_ap: float,
#     test_ar: float,
#     test_ap: float,
#     optim: Optimizer = None,
# ) -> TrainingInfo:
#     current_time_string = datetime.now().strftime("%m-%d-%Y %H-%M-%S")

#     model_path = (
#         (
#             f"val_ar_{val_ar:.4f}_ap_{val_ap:.4f}_"
#             + f"test_ar_{test_ar:.4f}_ap_{test_ap:.4f}_"
#             + f"epoch{train_info.epoch}_{train_info.clinical_cond}Clincal_{current_time_string}"
#             + f"_{train_info.model_setup.name}"
#         )
#         .replace(":", "_")
#         .replace(".", "_")
#     )

#     train_info.final_model_path = model_path

#     torch.save(
#         model.state_dict(),
#         os.path.join(os.path.join("trained_models", train_info.final_model_path)),
#     )

#     # Save optimizer if necessary.
#     if optim:
#         torch.save(
#             optim.state_dict(),
#             os.path.join(
#                 os.path.join("trained_models", f"{train_info.final_model_path}_optim")
#             ),
#         )

#     with open(
#         os.path.join("training_records", f"{train_info.final_model_path }.pkl"), "wb",
#     ) as train_info_f:
#         pickle.dump(train_info, train_info_f)

#     return train_info


def save_checkpoint(
    train_info: TrainingInfo,
    model: nn.Module,
    val_ar: float,
    val_ap: float,
    test_ar: float,
    test_ap: float,
    optimizer: Optimizer = None,
    dynamic_weight: nn.Module = None,
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

    saving_dict = {
        "model_state_dict":model.state_dict()
        
    }
    if optimizer:
        saving_dict["optimizer_state_dict"] = optimizer.state_dict()

    if dynamic_weight:
        saving_dict["dynamic_weight_state_dict"] =dynamic_weight.state_dict()

    torch.save(
        saving_dict,
        os.path.join(os.path.join("trained_models", train_info.final_model_path)),
    )

    # saving the train_info.
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
    setup,
    train_info: TrainingInfo,
    val_ap_ar,
    eval_params_dict: Dict,
    model: nn.Module,
    optim: Optimizer,
    test_dataloader: DataLoader,
    test_coco: Dataset,
    iou_types: List[str],
    device: str,
    score_thres: Dict[str, float] = None,
    dynamic_weight: nn.Module =None, 
    test_ap_ar = None,
) -> Tuple[float, float, TrainingInfo]:
    
    val_ar, val_ap = val_ap_ar['ar'],  val_ap_ar['ap']

    ## Targeting the model with higher Average Recall and Average Precision.
    if val_ar > train_info.best_val_ar or val_ap > train_info.best_val_ap:
        if test_ap_ar is None:
            train_info.test_evaluator, test_logger = xami_evaluate(
                setup=setup,
                model=model,
                data_loader=test_dataloader,
                device=device,
                params_dict=eval_params_dict,
                coco=test_coco,
                iou_types=iou_types,
                score_thres=score_thres,
            )
            test_ap_ar = get_ap_ar(train_info.test_evaluator)

        if val_ar > train_info.best_val_ar:
            ## Save best validation model
            previous_ar_model = deepcopy(train_info.best_ar_val_model_path)
            train_info = save_checkpoint(
                train_info=train_info,
                model=model,
                val_ar=val_ar,
                val_ap=val_ap,
                test_ar=test_ap_ar['ar'],
                test_ap=test_ap_ar['ap'],
                optimizer=optim,
                dynamic_weight=dynamic_weight
            )
            train_info.best_ar_val_model_path = train_info.final_model_path
            train_info.best_val_ar = val_ar

            if not train_info.still_has_path(previous_ar_model):
                remove_previous_model(previous_ar_model)
                train_info.removed_model_paths.append(previous_ar_model)

        if val_ap > train_info.best_val_ap:
            previous_ap_model = deepcopy(train_info.best_ap_val_model_path)
            train_info = save_checkpoint(
                train_info=train_info,
                model=model,
                val_ar=val_ar,
                val_ap=val_ap,
                test_ar=test_ap_ar['ar'],
                test_ap=test_ap_ar['ap'],
                optimizer=optim,
                dynamic_weight=dynamic_weight,
            )
            train_info.best_ap_val_model_path = train_info.final_model_path
            train_info.best_val_ap = val_ap
            
            if not train_info.still_has_path(previous_ap_model):
                remove_previous_model(previous_ap_model)
                train_info.removed_model_paths.append(previous_ap_model)

    # we should check existence of the model: 
    model_save_checking(train_info)

    return val_ar, val_ap, train_info


def model_save_checking(train_info: TrainingInfo):
    # path checking 
    if not train_info.best_ap_val_model_path is None:
        exist = model_path_checking_existence(train_info.best_ap_val_model_path)
        if not exist:
            raise FileNotFoundError(f"The best AP model file is not found: [{train_info.best_ap_val_model_path}]")
    
    if not train_info.best_ar_val_model_path is None:
        exist = model_path_checking_existence(train_info.best_ar_val_model_path)
        if not exist:
            raise FileNotFoundError(f"The best AR model file is not found: [{train_info.best_ar_val_model_path}]")
    
    if not train_info.final_model_path is None:
        exist = model_path_checking_existence(train_info.best_ar_val_model_path)
        if not exist:
            raise FileNotFoundError(f"The final model file is not found: [{train_info.final_model_path}]")

def model_path_checking_existence(model_path):
    # check the model exist:
    return os.path.exists(os.path.join(os.path.join("trained_models", model_path))) and os.path.exists(os.path.join("training_records", f"{model_path}.pkl"))




def end_train(
    setup,
    train_info: TrainingInfo,
    model: nn.Module,
    optim: Optimizer,
    eval_params_dict: Dict,
    last_val_ar: float,
    last_val_ap: float,
    test_dataloader: DataLoader,
    device: str,
    test_coco: Dataset,
    iou_types: List[str],
    score_thres: Dict[str, float] = None,
    dynamic_weight: nn.Module = None,
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

    train_info.test_evaluator, test_logger = xami_evaluate(
        setup= setup,
        model=model,
        data_loader=test_dataloader,
        device=device,
        params_dict=eval_params_dict,
        coco=test_coco,
        iou_types=iou_types,
        score_thres=score_thres,
        
    )

    test_ap_ar = get_ap_ar(train_info.test_evaluator)

    train_info = save_checkpoint(
        train_info=train_info,
        model=model,
        val_ar=last_val_ar,
        val_ap=last_val_ap,
        test_ar=test_ap_ar['ar'],
        test_ap=test_ap_ar['ap'],
        optimizer=optim,
        dynamic_weight = dynamic_weight,
    )

    print_f.print_title(
        f"The final model has been saved to: [{train_info.final_model_path}]"
    )
    return train_info

