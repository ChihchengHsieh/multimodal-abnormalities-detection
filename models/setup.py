from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelSetup:

    # which mode of dataset is used during training.
    # [normal] - the images with same dicom_id will be seen as different instances.
    # [unified] - the images with same dicom_id will be seen as only one instance.
    dataset_mode: str = "normal" # [normal, unified]

    # name of the model.
    name: str = None

    # setting up is fixation mask will passed into the model.
    use_fixations: bool = False

    # this will save the model with best validation performance across each epochs.
    save_early_stop_model: bool  = True

    # Will the training process will be recorded. (The TrainInfo instance will be saved with the weights of model.)
    record_training_performance: bool = True

    # define the backbone used in the model. 
    # If fixation is used, then both image and fixation backbones will use this architecture.
    backbone: str = "mobilenet_v3"  # [resnet18, resnet50, swin, mobilenet_v3] 

    # optimiser for training the model, SGD is default for training CNN.
    optimiser: str = "sgd"  # [adamw, sgd]

    # learning rate.
    lr: float = 0.0005

    # L2 regulariser
    weight_decay: float = 0.05

    #####################
    # Pretrained setup.
    #####################

    # if the image backbone is pretrained.
    image_backbone_pretrained: bool = True
    # if the fixation backbone is pretrained.
    fixation_backbone_pretrained: bool = False

    image_size: int = 256
    backbone_out_channels: int = 64
    batch_size: int = 4
    warmup_epochs: int = 0

    lr_scheduler: str = "ReduceLROnPlateau"  # [ReduceLROnPlateau, MultiStepLR]

    reduceLROnPlateau_factor: float = 0.1
    reduceLROnPlateau_patience: int = 3
    reduceLROnPlateau_full_stop: bool = False

    multiStepLR_milestones: List[int] = field(default_factory=lambda: [30, 50, 70, 90])
    multiStepLR_gamma: float = 0.1
    
    ## For warming up the training, but found not useful in our case.
    # warmup_epoch: int = 10
    # warmup_factor: float = 1.0 / 1000;


    #######################
    # Model related params
    #######################
    
    representation_size: int = 1024
    mask_hidden_layers: int = 256

    using_fpn: bool = False # the fpn is only implemented for ResNet and SwinTranformer.
    use_mask: bool = True

    fuse_conv_channels: int = 32

    box_head_dropout_rate: float = 0
    fuse_depth: int = 4

    fusion_strategy: str = "concat"  # ["add", "concat"]
    fusion_residule: bool = False

    gt_in_train_till: int = 20

    measure_test: bool = True

