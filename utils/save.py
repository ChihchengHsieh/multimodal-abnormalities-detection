import os
import torch
import pickle

from collections import OrderedDict
from datetime import datetime


def get_train_data(loger):
    train_data = {}
    for k in loger.meters.keys():
        train_data[k] = loger.meters[k].avg

    return train_data


def save_model(
    epoch,
    model,
    val_ar,
    val_ap,
    test_ar,
    test_ap,
    clinical_cond,
    train_logers,
    val_evaluators,
    test_evaluator,
):
    current_time_string = datetime.now().strftime("%m-%d-%Y %H-%M-%S")

    model_path = (
        (
            f"val_ar_{val_ar:.4f}_ap_{val_ap:.4f}_"
            + f"test_ar_{test_ar:.4f}_ap_{test_ap:.4f}_"
            + f"epoch{epoch}_{clinical_cond}Clincal_{current_time_string}"
        )
        .replace(":", "_")
        .replace(".", "_")
    )

    torch.save(
        model.state_dict(), os.path.join(os.path.join("trained_models", model_path)),
    )

    training_record = OrderedDict(
        {
            "train_data": [get_train_data(loger) for loger in train_logers],
            "val_evaluators": val_evaluators,
            "test_evaluator": test_evaluator,
        }
    )

    with open(
        os.path.join("training_records", f"{model_path}.pkl"), "wb",
    ) as training_record_f:
        pickle.dump(training_record, training_record_f)

    return model_path


def load_model(
    model, model_path, device,
):
    model.load_state_dict(
        torch.load(os.path.join("trained_models", model_path), map_location=device)
    )

    with open(os.path.join("training_records", f"{model_path}.pkl"), 'rb') as f:
        training_record = pickle.load(f)

    return model, training_record