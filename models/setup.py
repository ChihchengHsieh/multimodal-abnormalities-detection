from dataclasses import dataclass

@dataclass
class ModelSetup:
    use_clinical: bool
    use_custom_model: bool
    use_early_stop_model: bool
    name: str = None
    best_ar_val_model_path: str = None
    best_ap_val_model_path: str = None
    final_model_path: str = None
    backbone: str = "resnet50"  # [resnet50, swin]
    optimiser: str = "adamw"  # [adamw, sgd]
    lr: float = 0.0005
    weight_decay: float = 0.05
    pretrained: bool = False
    record_training_performance: bool = False
    dataset_mode: str = "unified"

