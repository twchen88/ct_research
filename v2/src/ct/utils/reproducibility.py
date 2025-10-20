import random, os
import numpy as np

"""
src/utils/reproducibility.py
----------------------------
This module provides functions to set the random seed for reproducibility in experiments.
"""


def set_global_seed(seed: int = 42):
    """
    Set the global random seed for reproducibility.

    Parameters:
        seed (int): The seed value to set. Default to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # For PyTorch, if used
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass