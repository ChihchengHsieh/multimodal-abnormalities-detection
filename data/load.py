import torch, random
import numpy as np

from typing import Dict, Tuple
from .datasets import ReflacxDataset, collate_fn
from .transforms import get_transform
from torch.utils.data import DataLoader

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader_g(seed: int = 0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def get_datasets(
    dataset_params_dict: Dict,
) -> Tuple[ReflacxDataset, ReflacxDataset, ReflacxDataset, ReflacxDataset]:

    detect_eval_dataset = ReflacxDataset(
        **{**dataset_params_dict,}, # , "dataset_mode": "unified"
        transforms=get_transform(train=False),
    )

    train_dataset = ReflacxDataset(
        **dataset_params_dict, split_str="train", transforms=get_transform(train=True), 
    )

    val_dataset = ReflacxDataset(
        **dataset_params_dict, split_str="val", transforms=get_transform(train=False),
    )

    test_dataset = ReflacxDataset(
        **dataset_params_dict, split_str="test", transforms=get_transform(train=False),
    )

    return detect_eval_dataset, train_dataset, val_dataset, test_dataset


def get_dataloaders(
    train_dataset: ReflacxDataset,
    val_dataset: ReflacxDataset,
    test_dataset: ReflacxDataset,
    batch_size: int = 4,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=get_dataloader_g(seed),
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=get_dataloader_g(seed),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=get_dataloader_g(seed),
    )

    return train_dataloader, val_dataloader, test_dataloader
