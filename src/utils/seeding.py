"""
Seeding and determinism utilities for PyTorch, NumPy, and Python random.

Follows project conventions (§5):
- Global seeding function
- PyTorch DataLoader worker_init_fn for per-worker RNG independence
"""

import random
import torch
import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Set global random seeds across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    """
    Worker initialization function for PyTorch DataLoader.
    Ensures per-worker RNG independence using base seed and worker ID.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        base_seed = getattr(dataset, "seed", 42)
        epoch = getattr(dataset, "epoch", 0)
        num_workers = worker_info.num_workers
        worker_seed = base_seed + epoch * num_workers + worker_id
        
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32 - 1))
        dataset.rng = np.random.default_rng(worker_seed)
