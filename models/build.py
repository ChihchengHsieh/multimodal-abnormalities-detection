import torch, torchvision

from typing import List
import torch.nn as nn
from .setup import ModelSetup
from .backbones.swin import FPN, BackboneWithFPN, SwinTransformer
from .detectors.rcnn import MultimodalMaskRCNN

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    AnchorGenerator,
)

MODEL_URLS = {
    "maskrcnn_resnet50_fpn_coco": "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
}

class NoAction(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x):
        return x


def create_model_from_setup(
    labels_cols: List[str], setup: ModelSetup, **kwargs
) -> nn.Module:
    if setup.use_custom_model:
        print("Load custom model")
        model = get_multimodal_rcnn_model(
            setup=setup, num_classes=len(labels_cols) + 1, **kwargs,
        )
    else:
        print("Load original model.")
        model = get_original_model_maskrcnn_resnet50_fpn(len(labels_cols) + 1,)

    return model


def get_multimodal_rcnn_model(
    num_classes, setup: ModelSetup, mask_hidden_layers=256, **kwargs,
):
    if setup.using_fpn:
        # Feature Pyramid Network: https://arxiv.org/abs/1612.03144v2)
        if setup.backbone.startswith("resnet"):
            print("Using ResNet as backbone")
            model = multimodal_maskrcnn_resnet_fpn(
                setup=setup, pretrained_backbone=setup.pretrained, **kwargs,
            )

        elif setup.backbone == "swin":
            print("Using SwinTransformer as backbone")
            model = multimodal_maskrcnn_swin_fpn(**kwargs,)
        else:
            raise Exception(f"Unsupported FPN backbone {setup.backbone}")

    else:
        model = multimodal_maskrcnn_with_backbone(
            setup=setup, pretrained_backbone=setup.pretrained, **kwargs,
        )

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    if setup.use_mask:
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

        # and replace the mask predictor with a new one
        print(f"Mask Hidden Layers {mask_hidden_layers}")
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, mask_hidden_layers, num_classes
        )

    return model

def multimodal_maskrcnn_resnet_fpn(
    setup: ModelSetup,
    pretrained=False,
    pretrained_backbone=True,
    progress=True,
    num_classes=91,
    trainable_backbone_layers=None,
    # backbone_out_channels=64,
    **kwargs,
):
    trainable_backbone_layers = torchvision.models.detection.backbone_utils._validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3
    )

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False

    if pretrained_backbone:
        print(f"Using pretrained backbone. {setup.backbone}")
    else:
        print("Not using pretrained backbone.")

    backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
        setup.backbone, pretrained_backbone, trainable_layers=trainable_backbone_layers
    )

    clinical_backbone = None

    if setup.spatialise_clinical and setup.spatialise_method == "convs": 
        clinical_backbone = get_clinical_backbone(setup)

    model = MultimodalMaskRCNN(
        setup, backbone, num_classes, clinical_backbone=clinical_backbone, **kwargs,
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

    clinical_backbone = None
    if setup.spatialise_clinical and setup.spatialise_method == "convs":
        clinical_backbone = get_clinical_backbone(
            setup, fpn_args=fpn_args, swin_args=swin_args
        )

    model = MultimodalMaskRCNN(
        setup,
        backbone,
        num_classes,
        clinical_backbone=clinical_backbone,
        **kwargs,
    )
    return model


def remove_last(model):
    if hasattr(model, "features"):
        return model.features

    elif hasattr(model, "fc"):
        model.fc = NoAction()
        if hasattr(model, "avgpool"):
            model.avgpool = NoAction()
        return model

    elif hasattr(model, "classifier"):
        model.classifier = NoAction()
        if hasattr(model, "avgpool"):
            model.avgpool = NoAction()
        return model

def to_feature_extract_backbone(resnet):
    return nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
        resnet.layer4,
    )

def get_normal_backbone(
    setup: ModelSetup, pretrained_backbone=True,
):
    if setup.backbone == "resnet18":
        backbone = to_feature_extract_backbone(
            torchvision.models.resnet18(pretrained=pretrained_backbone)
        )
        backbone.out_channels = 512
    elif setup.backbone == "resnet50":
        backbone = to_feature_extract_backbone(
            torchvision.models.resnet50(pretrained=pretrained_backbone)
        )
        backbone.out_channels = 512
    elif setup.backbone == "mobilenet_v2":
        backbone = remove_last(
            torchvision.models.mobilenet_v2(pretrained=pretrained_backbone)
        )
        backbone.out_channels = 1280
    elif setup.backbone == "mobilenet_v3":
        backbone = remove_last(
            torchvision.models.mobilenet_v3_small(pretrained=pretrained_backbone)
        )
        backbone.out_channels = 576
    elif setup.backbone == "custom1":
        resnet = torchvision.models.resnet18(pretrained=pretrained_backbone)
        backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # 64
            # resnet.layer2, # 128
            # resnet.layer3, # 256
            # resnet.layer4, # 512
        )
        backbone.out_channels = 64
    elif setup.backbone == "custom2":
        resnet = torchvision.models.resnet18(pretrained=pretrained_backbone)
        backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # 64
            resnet.layer2,  # 128
            # resnet.layer3, # 256
            # resnet.layer4, # 512
        )
        backbone.out_channels = 128
    elif setup.backbone == "custom3":
        resnet = torchvision.models.resnet18(pretrained=pretrained_backbone)
        backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # 64
            resnet.layer2,  # 128
            resnet.layer3,  # 256
            # resnet.layer4, # 512
        )
        backbone.out_channels = 256
    else:
        raise Exception(f"Unsupported backbone {setup.backbone}")

    if setup.backbone_out_channels:
        backbone = nn.Sequential(
            backbone,
            nn.Conv2d(backbone.out_channels, setup.backbone_out_channels, 3, 1, 1),
        )
        backbone.out_channels = setup.backbone_out_channels

    if pretrained_backbone:
        print(f"Using pretrained backbone. {setup.backbone}")
    else:
        print("Not using pretrained backbone.")

    return backbone

def get_clinical_backbone(
    setup: ModelSetup, fpn_args=None, swin_args=None,
):
    if setup.using_fpn:
        if setup.backbone.startswith("resnet"):
            print("Using ResNet as clinical backbone")
            trainable_backbone_layers = torchvision.models.detection.backbone_utils._validate_trainable_layers(
                setup.pretrained, None, 5, 3
            )
            clinical_backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                setup.backbone,
                setup.pretrained,
                trainable_layers=trainable_backbone_layers,
            )

        elif setup.backbone == "swin":
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

            clinical_backbone = BackboneWithFPN(
                backbone=SwinTransformer(**swin_args), fpn=FPN(**fpn_args),
            )
        else:
            raise Exception(f"Unsupported FPN backbone {setup.backbone}")

    else:
        clinical_backbone = get_normal_backbone(
            pretrained_backbone=setup.pretrained, setup=setup,
        )

    return clinical_backbone

def multimodal_maskrcnn_with_backbone(
    setup: ModelSetup,
    pretrained_backbone=True,
    num_classes=91,
    **kwargs,
):
    backbone = get_normal_backbone(setup=setup, pretrained_backbone=pretrained_backbone)
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"], output_size=7, sampling_ratio=2
    )

    clinical_backbone = None
    if setup.spatialise_clinical and setup.spatialise_method == "convs":
        clinical_backbone = get_clinical_backbone(setup)

    model = MultimodalMaskRCNN(
        setup,
        backbone,
        num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        clinical_backbone = clinical_backbone,
        **kwargs,
    )

    return model


def get_original_model_maskrcnn_resnet50_fpn(
    num_classes,
    rpn_nms_thresh=0.3,
    box_detections_per_img=10,
    box_nms_thresh=0.2,
    rpn_score_thresh=0.0,
    box_score_thresh=0.05,
    **kwargs,
):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=False,
        rpn_nms_thresh=rpn_nms_thresh,
        box_detections_per_img=box_detections_per_img,
        box_nms_thresh=box_nms_thresh,
        rpn_score_thresh=rpn_score_thresh,
        box_score_thresh=box_score_thresh,
        **kwargs,
    )

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model
