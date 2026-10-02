import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG used in this project and make cuDNN deterministic."""
    random.seed(seed)                  # data split, teacher-forcing coin
    np.random.seed(seed)               # any NumPy shuffling
    torch.manual_seed(seed)            # weight init, DataLoader shuffle (CPU)
    torch.cuda.manual_seed_all(seed)   # all GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
