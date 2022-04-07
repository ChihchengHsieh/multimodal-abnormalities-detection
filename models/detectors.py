from .swin import FPN, SwinFPN, SwinTransformer
from .rcnn import MultimodalMaskRCNN

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

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


