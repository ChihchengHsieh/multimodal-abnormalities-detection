from enum import Enum
from io import UnsupportedOperation
from turtle import back

from .detectors import multimodal_maskrcnn_swin_fpn
from .rcnn import multimodal_maskrcnn_resnet50_fpn

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_multimodal_model(
    num_classes,
    setup,
    **kwargs,
):
    if setup.backbone ==  'resnet50':
        print("Using ResNet50 as backbone")
        model = multimodal_maskrcnn_resnet50_fpn(
            pretrained=setup.pretrained,
            **kwargs,
        )

    elif setup.backbone == 'swin':
        print("Using SwinTransformer as backbone")
        model = multimodal_maskrcnn_swin_fpn(
            **kwargs,
        )
    
    else:
        raise Exception(f"Unsupported backbone {setup.backbone}")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model
