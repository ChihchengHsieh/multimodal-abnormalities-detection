from typing import List
import numpy as np
from sklearn.model_selection import train_test_split


def get_split_string(
    train_idxs: List[int], val_idxs: List[int], test_idxs: List[int], idx: int
) -> str:

    if idx in train_idxs:
        return "train"

    if idx in val_idxs:
        return "val"

    if idx in test_idxs:
        return "test"

    raise Exception(f"Index {idx} not in any split.")


def get_split_list(
    split_len: int, train_portion=0.8, val_portion=0.5, seed=0
) -> List[str]:

    split_idxs = list(range(split_len))

    train_idxs, val_test_idxs = train_test_split(
        split_idxs, train_size=train_portion, random_state=seed, shuffle=True
    )

    val_idxs, test_idxs = train_test_split(
        val_test_idxs, train_size=val_portion, random_state=seed, shuffle=True
    )

    return [
        get_split_string(train_idxs, val_idxs, test_idxs, i) for i in range(split_len)
    ]

