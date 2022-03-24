import torch, os

from http.client import ImproperConnectionState
from models.rcnn import get_multimodal_model_instance_segmentation, get_model_instance_segmentation

model_path_dict = {

    "20epoch": {
        "original": {
            "with_clinical": None,
            "without_clinical": "val_ar_0_2536_ap_0_1206_test_ar_0_2864_ap_0_1256_epoch20_WithoutClincal_03-13-2022 23-20-30",
        },
        "custom": {
            "with_clinical": "val_ar_0_3212_ap_0_1481_test_ar_0_2996_ap_0_1704_epoch20_WithClincal_03-14-2022 01-45-59",
            "without_clinical": "val_ar_0_3397_ap_0_1889_test_ar_0_3146_ap_0_1402_epoch20_WithoutClincal_03-14-2022 00-30-30",
        },
    },
    "early_stop": {
        "custom": {
            "with_clinical": "val_ar_0_3909_ap_0_1828_test_ar_0_3808_ap_0_1542_epoch39_WithClincal_03-14-2022 17-14-52",
            "without_clinical": "val_ar_0_3410_ap_0_1753_test_ar_0_3553_ap_0_1832_epoch33_WithoutClincal_03-14-2022 19-10-27",
        },
        "original": {
            "with_clinical": None,
            "without_clinical": "val_ar_0_3000_ap_0_1542_test_ar_0_3504_ap_0_1386_epoch15_WithoutClincal_03-14-2022 03-10-11",
        },
    },
}


def get_model_path(use_early_stop_model, use_custom_modal, use_clinical):
    model_path = model_path_dict[
        "early_stop" if use_early_stop_model else "20epoch"
    ]["custom" if use_custom_modal else "original"][
        "with_clinical" if use_clinical else "without_clinical"
    ]

    return model_path


def get_trained_model(
    dataset, use_early_stop_model, use_custom_modal, use_clinical, device,
):
    model_path = get_model_path(use_early_stop_model, use_custom_modal, use_clinical)

    if use_custom_modal:
        model = get_multimodal_model_instance_segmentation(
            len(dataset.labels_cols) + 1, use_clinical=use_clinical,
        )

    else:
        model = get_model_instance_segmentation(
            len(dataset.labels_cols) +1,
        )

    model.to(device)
    model.load_state_dict(
        torch.load(os.path.join("trained_models", model_path), map_location=device)
    )

    return model, model_path