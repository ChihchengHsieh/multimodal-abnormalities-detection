
import torch, gc, random

import numpy as np

def clean_memory_get_device():
    gc.collect()
    # torch.cuda.memory_summary(device=None, abbreviated=False)

    use_gpu = torch.cuda.is_available()
    device = 'cuda' if use_gpu else 'cpu'
    print(f"This notebook will running on device: [{device.upper()}]")

    if use_gpu:
        torch.cuda.empty_cache()
    return device


def reproducibility(seed=0):
    torch.manual_seed(seed)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloader_g(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g