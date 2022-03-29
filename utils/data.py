import torch

from data.dataset import ReflacxDataset, collate_fn
from utils.transforms import get_transform

def get_datasets(dataset_params_dict):

    detect_eval_dataset = ReflacxDataset(
        **{**dataset_params_dict, "dataset_mode": "unified",},
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


def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=4):

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
    )

    return train_dataloader, val_dataloader, test_dataloader