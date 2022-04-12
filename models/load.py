from genericpath import isfile
import os, pickle, torch
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from .build import get_multimodal_model
from utils.train import get_optimiser

from .rcnn import get_model_instance_segmentation


@dataclass
class ModelSetup:
    use_clinical: bool
    use_custom_model: bool
    use_early_stop_model: bool
    name: str = None
    best_ar_val_model_path: str = None
    best_ap_val_model_path: str = None
    final_model_path: str = None
    backbone: str = "resnet50"  # [resnet50, swin]
    optimiser: str = "adamw"  # [adamw, sgd]
    lr: float = 0.0005
    weight_decay: float = 0.05
    pretrained: bool = False
    record_training_performance: bool = False
    dataset_mode: str = "unified"


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


# class TrainedModels(Enum):
#     original = "val_ar_0_5230_ap_0_2576_test_ar_0_5678_ap_0_2546_epoch28_WithoutClincal_03-28-2022 06-56-13_original"

#     custom_without_clinical = "val_ar_0_4575_ap_0_2689_test_ar_0_4953_ap_0_2561_epoch40_WithoutClincal_03-28-2022 09-15-40_custom_without_clinical"

#     custom_with_clinical_drop0 = "val_ar_0_5363_ap_0_2963_test_ar_0_5893_ap_0_2305_epoch36_WithClincal_03-28-2022 20-06-43_custom_with_clinical"

#     custom_with_clinical_drop2 = "val_ar_0_5126_ap_0_2498_test_ar_0_5607_ap_0_2538_epoch18_WithClincal_03-28-2022 10-18-55_custom_with_clinical"

#     custom_with_clinical_drop3 = "val_ar_0_3993_ap_0_2326_test_ar_0_4957_ap_0_2390_epoch50_WithClincal_03-28-2022 16-06-00_custom_with_clinical"

#     custom_with_clinical_drop5 = "val_ar_0_4955_ap_0_2942_test_ar_0_5449_ap_0_2566_epoch28_WithClincal_03-28-2022 17-25-34_custom_with_clinical"

#     overfitting = "val_ar_0_2113_ap_0_1818_test_ar_0_2767_ap_0_1532_epoch250_WithClincal_03-31-2022 23-09-38_custom_with_clinical"


class TrainedModels(Enum):

    # custom_without_clinical_pretrained
    custom_without_clinical_pretrained_ar = "val_ar_0_5181_ap_0_2127_test_ar_0_5659_ap_0_2037_epoch36_WithoutClincal_04-10-2022 08-07-20_custom_without_clinical_pretrained"
    custom_without_clinical_pretrained_ap = "val_ar_0_4635_ap_0_2291_test_ar_0_5178_ap_0_1937_epoch17_WithoutClincal_04-10-2022 07-30-34_custom_without_clinical_pretrained"
    custom_without_clinical_pretrained_final = "val_ar_0_4279_ap_0_2052_test_ar_0_4103_ap_0_1757_epoch50_WithoutClincal_04-10-2022 08-34-20_custom_without_clinical_pretrained"

    # custom_with_clinical_pretrained
    custom_with_clinical_pretrained_ar = "val_ar_0_5171_ap_0_2336_test_ar_0_5267_ap_0_1545_epoch39_WithClincal_04-10-2022 10-15-22_custom_with_clinical_pretrained"
    custom_with_clinical_pretrained_ap = "val_ar_0_4581_ap_0_2496_test_ar_0_5533_ap_0_1655_epoch49_WithClincal_04-10-2022 10-38-54_custom_with_clinical_pretrained"
    custom_with_clinical_pretrained_final = "val_ar_0_3912_ap_0_1756_test_ar_0_5129_ap_0_1736_epoch50_WithClincal_04-10-2022 10-41-48_custom_with_clinical_pretrained"

    # custom_without_clinical_no_pretrained
    custom_without_clinical_no_pretrained_ar = "val_ar_0_4295_ap_0_1586_test_ar_0_4991_ap_0_2054_epoch39_WithoutClincal_04-10-2022 12-02-33_custom_without_clinical_no_pretrained"
    custom_without_clinical_no_pretrained_ap = "val_ar_0_4156_ap_0_1726_test_ar_0_5085_ap_0_1841_epoch40_WithoutClincal_04-10-2022 12-05-11_custom_without_clinical_no_pretrained"
    custom_without_clinical_no_pretrained_final = "val_ar_0_2427_ap_0_1189_test_ar_0_3630_ap_0_1724_epoch50_WithoutClincal_04-10-2022 12-25-02_custom_without_clinical_no_pretrained"

    # custom_with_clinical_no_pretrained
    custom_with_clinical_no_pretrained_ar = "val_ar_0_4338_ap_0_2249_test_ar_0_4445_ap_0_1816_epoch49_WithClincal_04-10-2022 14-24-02_custom_with_clinical_no_pretrained"
    custom_with_clinical_no_pretrained_ap = "val_ar_0_4338_ap_0_2249_test_ar_0_4445_ap_0_1816_epoch49_WithClincal_04-10-2022 14-24-04_custom_with_clinical_no_pretrained"
    custom_with_clinical_no_pretrained_final = "val_ar_0_3519_ap_0_1750_test_ar_0_4113_ap_0_1602_epoch50_WithClincal_04-10-2022 14-27-02_custom_with_clinical_no_pretrained"

    # custom_without_clinical_swim
    custom_without_clinical_swim_ar = "val_ar_0_4324_ap_0_1400_test_ar_0_4371_ap_0_1408_epoch45_WithoutClincal_04-10-2022 16-09-22_custom_with_clinical_swim"
    custom_without_clinical_swim_ap = "val_ar_0_3653_ap_0_1552_test_ar_0_3867_ap_0_1298_epoch29_WithoutClincal_04-10-2022 15-34-21_custom_with_clinical_swim"
    custom_without_clinical_swim_final = "val_ar_0_3816_ap_0_1417_test_ar_0_3788_ap_0_1313_epoch50_WithoutClincal_04-10-2022 16-20-24_custom_with_clinical_swim"

    # custom_with_clinical_swim
    custom_with_clinical_swim_ar = "val_ar_0_4182_ap_0_1406_test_ar_0_4256_ap_0_0967_epoch44_WithClincal_04-10-2022 18-17-49_custom_with_clinical_swim"
    custom_with_clinical_swim_ap = "val_ar_0_3589_ap_0_1554_test_ar_0_4126_ap_0_1312_epoch41_WithClincal_04-10-2022 18-09-37_custom_with_clinical_swim"
    custom_with_clinical_swim_final = "val_ar_0_3008_ap_0_0923_test_ar_0_3878_ap_0_1092_epoch50_WithClincal_04-10-2022 18-33-30_custom_with_clinical_swim"


def create_model_from_setup(labels_cols, setup: ModelSetup, **kwargs):
    if setup.use_custom_model:
        print("Load custom model")
        model = get_multimodal_model(
            setup=setup,
            num_classes=len(labels_cols) + 1,
            use_clinical=setup.use_clinical,
            **kwargs,
        )
    else:
        print("Load original model.")
        model = get_model_instance_segmentation(len(labels_cols) + 1,)

    return model


def get_trained_model(
    model_select: TrainedModels, labels_cols, device, **kwargs,
) -> Tuple[Any, Optional[TrainingInfo]]:

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
        optim: torch.optim.optimizer.Optimizer = get_optimiser(train_info.model_setup)
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


get_current_epoch = lambda select_model: int(
    (
        [substr for substr in select_model.value.split("_") if "epoch" in substr][0]
    ).replace("epoch", "")
)

get_model_name = lambda select_model: str(select_model).split(".")[-1]
