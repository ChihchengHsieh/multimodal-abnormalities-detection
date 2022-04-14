import torch, torchvision

from typing import List
import torch.nn as nn
from .setup import ModelSetup
from .backbones.swin import FPN, SwinFPN, SwinTransformer
from .detectors.rcnn import MultimodalMaskRCNN

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

MODEL_URLS = {
    "maskrcnn_resnet50_fpn_coco": "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
}

def create_model_from_setup(labels_cols: List[str], setup: ModelSetup, **kwargs) -> nn.Module:
    if setup.use_custom_model:
        print("Load custom model")
        model = get_multimodal_rcnn_model(
            setup=setup,
            num_classes=len(labels_cols) + 1,
            use_clinical=setup.use_clinical,
            **kwargs,
        )
    else:
        print("Load original model.")
        model = get_original_model_maskrcnn_resnet50_fpn(len(labels_cols) + 1,)

    return model

def get_multimodal_rcnn_model(
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

def multimodal_maskrcnn_resnet50_fpn(
    pretrained=False,
    progress=True,
    num_classes=91,
    pretrained_backbone=True,
    trainable_backbone_layers=None,
    clinical_input_channels=32,
    clinical_num_len=9,
    clinical_conv_channels=256,
    fuse_conv_channels=256,
    use_clinical=True,
    **kwargs,
):
    trainable_backbone_layers = torchvision.models.detection.backbone_utils._validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3
    )

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False

    backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
        "resnet50", pretrained_backbone, trainable_layers=trainable_backbone_layers
    )

    model = MultimodalMaskRCNN(
        backbone,
        num_classes,
        clinical_input_channels=clinical_input_channels,
        clinical_num_len=clinical_num_len,
        clinical_conv_channels=clinical_conv_channels,
        fuse_conv_channels=fuse_conv_channels,
        use_clinical=use_clinical,
        **kwargs,
    )

    if pretrained:
        print("Using pretrained model")
        state_dict = torch.hub.load_state_dict_from_url(
            MODEL_URLS["maskrcnn_resnet50_fpn_coco"], progress=progress
        )
        model.load_state_dict(state_dict, strict=False)
        torchvision.models.detection._utils.overwrite_eps(model, 0.0)
    else:
        print("Not using pretrained model.")

    return model

def multimodal_maskrcnn_swin_fpn(
    num_classes=91,
    clinical_input_channels=32,
    clinical_num_len=9,
    clinical_conv_channels=256,
    fuse_conv_channels=256,
    use_clinical=True,
    fpn_args=None,
    swin_args=None,
    **kwargs,
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

    backbone = SwinFPN(swin=SwinTransformer(**swin_args), fpn=FPN(**fpn_args),)
    
    model = MultimodalMaskRCNN(
        backbone,
        num_classes,
        clinical_input_channels=clinical_input_channels,
        clinical_num_len=clinical_num_len,
        clinical_conv_channels=clinical_conv_channels,
        fuse_conv_channels=fuse_conv_channels,
        use_clinical=use_clinical,
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

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model