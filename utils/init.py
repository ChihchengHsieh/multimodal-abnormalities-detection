import torch, gc, random
import numpy as np


def clean_memory_get_device() -> str:
    gc.collect()
    # torch.cuda.memory_summary(device=None, abbreviated=False)

    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    print(f"This notebook will running on device: [{device.upper()}]")

    if use_gpu:
        torch.cuda.empty_cache()
    return device


def reproducibility(seed: int = 0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
