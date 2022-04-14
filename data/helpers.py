from typing import Dict
import torch
from collections import ChainMap

def map_target_to_device(target: Dict, device: str) -> Dict:
    key_value_pairs = []

    for k, v in target.items():
        if isinstance(v, torch.Tensor):
            key_value_pairs.append({k: v.to(device)})
        else:
            key_value_pairs.append({k: v})

    return dict(ChainMap(*key_value_pairs))