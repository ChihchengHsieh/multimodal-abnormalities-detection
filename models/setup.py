from dataclasses import dataclass, field
from typing import List

from data.constants import DEFAULT_MIMIC_CLINICAL_CAT_COLS, DEFAULT_MIMIC_CLINICAL_NUM_COLS


@dataclass
class ModelSetup:
    use_clinical: bool
    use_custom_model: bool
    use_early_stop_model: bool
    name: str = None
    backbone: str = "resnet50"  # [resnet18, resnet50, swin, mobilenet_v3]
    optimiser: str = "adamw"  # [adamw, sgd]
    lr: float = 0.0005
    weight_decay: float = 0.05
    pretrained: bool = False
    record_training_performance: bool = False
    dataset_mode: str = "normal" # [normal, unified]
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

    # warmup_epoch: int = 10
    # warmup_factor: float = 1.0 / 1000;

    ## Model related params:
    representation_size: int = 1024
    mask_hidden_layers: int = 256

    using_fpn: bool = False
    use_mask: bool = True

    clinical_input_channels: int = 32
    clinical_expand_conv_channels: int = 32

    including_clinical_num: List[str] =  field(default_factory = lambda: DEFAULT_MIMIC_CLINICAL_NUM_COLS)
    including_clinical_cat: List[str] = field(default_factory = lambda:DEFAULT_MIMIC_CLINICAL_CAT_COLS)

    clinical_conv_channels: int = 32

    fuse_conv_channels: int = 32

    box_head_dropout_rate: float = 0
    fuse_depth: int = 4

    spatialise_clinical: bool = True # 3d
    add_clinical_to_roi_heads: bool = False # 1d

    fusion_strategy: str = "concat"  # ["add", "concat"]
    fusion_residule: bool = False

    gt_in_train_till: int = 20

    spatialise_method: str = "convs"  # ["convs", "repeat"]
    normalise_clinical_num: bool = False
    measure_test: bool = False

    pre_spatialised_layer: int = None 

    # what that layer should be called?

    def has_categorical_clinical_features (self,):
        if not hasattr(self, "including_clinical_cat"):
            return True

        return not self.including_clinical_cat is None and len(self.including_clinical_cat) > 0
    
    def has_numerical_clinical_features(self,):
        if not hasattr(self, "including_clinical_num"):
            return True

        return not self.including_clinical_num is None and len(self.including_clinical_num) > 0

    def has_pre_spa(self):
        return hasattr(self, "pre_spatialised_layer") and not self.pre_spatialised_layer is None and self.pre_spatialised_layer > 0


    def get_input_dim_for_spa(self,):
        if not self.use_clinical:
            return 0

        if self.has_pre_spa() or self.has_categorical_clinical_features():
            return self.clinical_input_channels

        return self.get_clinical_num_len()

    def get_input_dim_for_pre_spa(self,):
        if not self.use_clinical:
            return 0

        if self.has_categorical_clinical_features():
            return self.clinical_input_channels

        return self.get_clinical_num_len()

    def get_clinical_num_len(self):
        return len(self.including_clinical_num) if hasattr(self, "including_clinical_num") else 9

    def get_clinical_cat_len(self):
        return len(self.including_clinical_cat) if hasattr(self, "including_clinical_cat") else 1

    def get_emb_dim(self,):
        return self.clinical_input_channels - self.get_clinical_num_len()

    


