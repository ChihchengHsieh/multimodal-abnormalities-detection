import os, pickle, torch
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from models.rcnn import (
    get_multimodal_model_instance_segmentation,
    get_model_instance_segmentation,
)


@dataclass
class ModelSetup:
    use_clinical: bool
    use_custom_model: bool
    use_early_stop_model: bool
    name: str = None
    best_ar_val_model_path: str = None
    best_ap_val_model_path: str = None
    final_model_path: str = None

class TrainingInfo:
    def __init__(self, model_setup: ModelSetup):
        self.train_data = []
        self.val_evaluators = []
        self.train_evaluators = []
        self.test_evaluator = None
        self.best_val_ar = -1
        self.best_val_ap = -1
        self.best_ar_val_model_path = None
        self.best_ap_val_model_path = None
        self.final_model_path = None
        self.previous_ar_model = None
        self.previous_ap_model = None
        self.model_setup = model_setup
        self.start_t = datetime.now()
        self.clinical_cond = "With" if model_setup.use_clinical else "Without"
        self.end_t = None
        self.epoch = 0
        super(TrainingInfo).__init__()

class TrainedModels(Enum):

    original = "val_ar_0_5230_ap_0_2576_test_ar_0_5678_ap_0_2546_epoch28_WithoutClincal_03-28-2022 06-56-13_original"

    custom_without_clinical = "val_ar_0_4575_ap_0_2689_test_ar_0_4953_ap_0_2561_epoch40_WithoutClincal_03-28-2022 09-15-40_custom_without_clinical"

    custom_with_clinical_drop0 = "val_ar_0_5363_ap_0_2963_test_ar_0_5893_ap_0_2305_epoch36_WithClincal_03-28-2022 20-06-43_custom_with_clinical"

    custom_with_clinical_drop2 = "val_ar_0_5126_ap_0_2498_test_ar_0_5607_ap_0_2538_epoch18_WithClincal_03-28-2022 10-18-55_custom_with_clinical"

    custom_with_clinical_drop3 = "val_ar_0_3993_ap_0_2326_test_ar_0_4957_ap_0_2390_epoch50_WithClincal_03-28-2022 16-06-00_custom_with_clinical"

    custom_with_clinical_drop5 = "val_ar_0_4955_ap_0_2942_test_ar_0_5449_ap_0_2566_epoch28_WithClincal_03-28-2022 17-25-34_custom_with_clinical"

    overfitting = "val_ar_0_2113_ap_0_1818_test_ar_0_2767_ap_0_1532_epoch250_WithClincal_03-31-2022 23-09-38_custom_with_clinical"


def create_model_from_setup(
    labels_cols, setup: ModelSetup,
):
    if setup.use_custom_model:
        model = get_multimodal_model_instance_segmentation(
            len(labels_cols) + 1, use_clinical=setup.use_clinical,
        )
    else:
        model = get_model_instance_segmentation(len(labels_cols) + 1,)

    return model


def get_trained_model(
    model_select: TrainedModels, labels_cols, device, include_train_info=False
) -> Tuple[Any, Optional[TrainingInfo]]:

    with open(os.path.join("training_records", f"{model_select.value}.pkl"), "rb") as f:
        train_info = pickle.load(f)

    model = create_model_from_setup(labels_cols, train_info.model_setup)
    model.to(device)

    model.load_state_dict(
        torch.load(os.path.join("trained_models", model_select.value), map_location=device)
    )

    if include_train_info:
        return model, train_info
    else:
        return model

