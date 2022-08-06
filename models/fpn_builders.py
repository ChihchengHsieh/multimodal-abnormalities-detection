from cProfile import label
import torch, torchvision

from typing import List
import torch.nn as nn

from .setup import ModelSetup
from .backbones.swin import FPN, BackboneWithFPN, SwinTransformer
from .detectors.rcnn import MultimodalMaskRCNN
from .backbones import get_normal_backbone

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

MODEL_URLS = {
    "maskrcnn_resnet50_fpn_coco": "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
}


def multimodal_maskrcnn_resnet_fpn(
    setup: ModelSetup,
    pretrained=False,
    progress=True,
    num_classes=91,
    trainable_backbone_layers=None,
    **kwargs,
):
    image_trainable_backbone_layers = torchvision.models.detection.backbone_utils._validate_trainable_layers(
        setup.image_backbone_pretrained, trainable_backbone_layers, 5, 3
    )

    fixation_trainable_backbone_layers = torchvision.models.detection.backbone_utils._validate_trainable_layers(
        setup.fixation_backbone_pretrained, trainable_backbone_layers, 5, 3
    )

    if setup.image_backbone_pretrained:
        print(f"Using pretrained backbone for images. {setup.backbone}")

    if setup.fixation_backbone_pretrained:
        print(f"Using pretrained backbone for fixations. {setup.backbone}")

    image_backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
        setup.backbone, setup.image_backbone_pretrained, trainable_layers=image_trainable_backbone_layers
    )

    fixation_backbone = (
        torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
            setup.backbone,
            setup.fixation_backbone_pretrained,
            trainable_layers=fixation_trainable_backbone_layers, # We train all the layers.
        )
        if setup.use_fixations
        else None
    )

    model = MultimodalMaskRCNN(
        setup, image_backbone, num_classes, fixation_backbone=fixation_backbone, **kwargs,
    )

    if pretrained:
        print("Using pretrained MaksRCNN model")
        state_dict = torch.hub.load_state_dict_from_url(
            MODEL_URLS["maskrcnn_resnet50_fpn_coco"], progress=progress
        )
        model.load_state_dict(state_dict, strict=False)
        torchvision.models.detection._utils.overwrite_eps(model, 0.0)
    else:
        print("Not using pretrained MaksRCNN model.")

    return model


def multimodal_maskrcnn_swin_fpn(
    setup: ModelSetup, num_classes=91, fpn_args=None, swin_args=None, **kwargs,
):
    if not fpn_args:
        fpn_args = {
            "in_channels": [96, 192, 384, 768],
            "out_channels": 256,
            "num_outs": 5,
        }

    if not swin_args:
        swin_args = {
            "pretrain_img_size": 256,
        }

    backbone = BackboneWithFPN(
        backbone=SwinTransformer(**swin_args), fpn=FPN(**fpn_args),
    )

    fixation_backbone = (
        BackboneWithFPN(backbone=SwinTransformer(**swin_args), fpn=FPN(**fpn_args),)
        if setup.use_fixations
        else None
    )

    model = MultimodalMaskRCNN(
        setup, backbone, num_classes, fixation_backbone==fixation_backbone, **kwargs,
    )
    return model
