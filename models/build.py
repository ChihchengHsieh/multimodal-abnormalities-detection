from cProfile import label
import torchvision

from typing import List

from models.backbones import get_normal_backbone
from models.fpn_builders import multimodal_maskrcnn_resnet_fpn, multimodal_maskrcnn_swin_fpn
from .setup import ModelSetup
from .detectors.rcnn import MultimodalMaskRCNN

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    AnchorGenerator,
)

def create_multimodal_rcnn_model(
    labels_cols: List[str], setup: ModelSetup, **kwargs,
):
    num_classes = len(labels_cols)

    if setup.using_fpn:
        # Feature Pyramid Network: (https://arxiv.org/abs/1612.03144v2), implementted for ResNet and SwinTranformer.
        if setup.backbone.startswith("resnet"):
            print("Using ResNet as backbone")
            model = multimodal_maskrcnn_resnet_fpn(
                setup=setup, **kwargs,
            )

        elif setup.backbone == "swin":
            print("Using SwinTransformer as backbone")
            model = multimodal_maskrcnn_swin_fpn(setup, **kwargs,)
        else:
            raise Exception(f"Unsupported FPN backbone {setup.backbone}")

    else:
        model = multimodal_maskrcnn_with_backbone(
            setup=setup, **kwargs,
        )

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    if setup.use_mask:
        print(f"{setup.name} will use mask, [{setup.mask_hidden_layers}] layers.")
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, setup.mask_hidden_layers, num_classes
        )

    return model

def multimodal_maskrcnn_with_backbone(
    setup: ModelSetup,
    num_classes=91,
    **kwargs,
):
    image_backbone = get_normal_backbone(setup=setup, pretrained_backbone=setup.image_backbone_pretrained)
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"], output_size=7, sampling_ratio=2
    )

    fixations_backbone = get_normal_backbone(setup=setup, pretrained_backbone=setup.fixation_backbone_pretrained) if setup.use_fixations else None

    model = MultimodalMaskRCNN(
        setup,
        image_backbone,
        num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        fixation_backbone = fixations_backbone,
        **kwargs,
    )

    return model
