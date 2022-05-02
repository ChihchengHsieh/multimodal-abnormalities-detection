from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelSetup:
    use_clinical: bool
    use_custom_model: bool
    use_early_stop_model: bool
    name: str = None
    best_ar_val_model_path: str = None
    best_ap_val_model_path: str = None
    final_model_path: str = None
    backbone: str = "resnet50"  # [resnet18, resnet50, swin]
    optimiser: str = "adamw"  # [adamw, sgd]
    lr: float = 0.0005
    weight_decay: float = 0.05
    pretrained: bool = False
    record_training_performance: bool = False
    dataset_mode: str = "unified"
    image_size: int = 256
    backbone_out_channels: int = 64
    batch_size: int = 4
    warmup_epochs: int = 0

    lr_scheduler: str = "ReduceLROnPlateau"  # [ReduceLROnPlateau, MultiStepLR]

    reduceLROnPlateau_factor: float = 0.1
    reduceLROnPlateau_patience: int = 3

    multiStepLR_milestones: List[int] = field(default_factory=lambda: [30, 50, 70, 90])
    multiStepLR_gamma: float = 0.1

    # warmup_epoch: int = 10
    # warmup_factor: float = 1.0 / 1000;

    ## Model related params:
    representation_size: int = 1024
    mask_hidden_layers: int = 256

    using_fpn: bool = False
    use_mask: bool = True

    clinical_expand_dropout_rate:float = 0
    clinical_conv_dropout_rate:float = 0
    clinical_input_channels:int =32
    clinical_num_len: int = 9
    clinical_conv_channels: int = 32

    fuse_conv_channels: int =32 
    fuse_dropout_rate:float = 0

    box_head_dropout_rate: float = 0
    fuse_depth: int = 4

