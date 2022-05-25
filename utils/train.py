import torch
import torch.nn as nn

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, MultiStepLR

from models.setup import ModelSetup


def get_optimiser(params, setup: ModelSetup) -> Optimizer:


    if setup.optimiser == "adamw":
        print(f"Using AdamW as optimizer with lr={setup.lr}")
        optimiser = torch.optim.AdamW(
            params, lr=setup.lr, betas=(0.9, 0.999), weight_decay=setup.weight_decay,
        )

    elif setup.optimiser == "sgd":
        print(f"Using SGD as optimizer with lr={setup.lr}")
        optimiser = torch.optim.SGD(
            params, lr=setup.lr, momentum=0.9, weight_decay=setup.weight_decay,
        )
    else:
        raise Exception(f"Unsupported optimiser {setup.optimiser}")

    return optimiser


def get_lr_scheduler(optimizer: Optimizer, setup: ModelSetup) -> _LRScheduler:

    if setup.lr_scheduler == "ReduceLROnPlateau":
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=setup.reduceLROnPlateau_factor,
            patience=setup.reduceLROnPlateau_patience,
            min_lr=1e-10,
        )
    elif setup.lr_scheduler == "MultiStepLR":
        lr_scheduler = MultiStepLR(
            optimizer,
            milestones=setup.multiStepLR_milestones,
            gamma=setup.multiStepLR_gamma,
        )
    else:
        lr_scheduler = None

    return lr_scheduler


def num_params(model):
    return sum([param.nelement() for param in model.parameters()])


def print_params_setup(model):
    print(f"[model]: {num_params(model):,}")
    print(f"[model.backbone]: {num_params(model.backbone):,}")
    print(f"[model.rpn]: {num_params(model.rpn):,}")
    print(f"[model.roi_heads]: {num_params(model.roi_heads):,}")
    print(f"[model.roi_heads.box_head]: {num_params(model.roi_heads.box_head):,}")
    print(
        f"[model.roi_heads.box_head.fc6]: {num_params(model.roi_heads.box_head.fc6):,}"
    )
    print(
        f"[model.roi_heads.box_head.fc7]: {num_params(model.roi_heads.box_head.fc7):,}"
    )
    print(
        f"[model.roi_heads.box_predictor]: {num_params(model.roi_heads.box_predictor):,}"
    )

    if hasattr(model.roi_heads, "mask_head") and not model.roi_heads.mask_head is None:
        print(f"[model.roi_heads.mask_head]: {num_params(model.roi_heads.mask_head):,}")

    if hasattr(model, "clinical_convs") and not model.clinical_convs is None:
        print(f"[model.clinical_convs]: {num_params(model.clinical_convs):,}")

    if hasattr(model, "fuse_convs") and not model.fuse_convs is None:
        print(f"[model.fuse_convs]: {num_params(model.fuse_convs):,}")