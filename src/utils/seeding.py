"""
Seeding and determinism utilities for PyTorch, NumPy, and Python random.

Follows project conventions (§5):
- Global seeding function
- PyTorch DataLoader worker_init_fn for per-worker RNG independence
"""

import random

import cv2
import numpy as np
import torch


def configure_cpu_threads(num_threads: int = 1) -> None:
    """Configure CPU thread allocation in the parent process before forking workers.

    Calling cv2.setNumThreads() inside a forked worker child process is undefined
    behaviour in OpenCV and causes worker segmentation faults. Setting it in the parent
    process ensures child processes inherit the single-thread setting safely across fork.
    """
    cv2.setNumThreads(num_threads)


def seed_everything(seed: int = 42) -> None:
    """Set global random seeds across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    configure_cpu_threads(1)


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker its own RNG stream.

    Without this, forked workers inherit identical RNG state and every worker
    composites the *same* samples: dataset variety silently collapses to 1/N, nothing
    errors, and the loss curve looks healthy while the model overfits far too fast
    (conventions §5).

    The loaders use `persistent_workers=True`, so this runs **once per worker per
    process**, not once per epoch. `dataset.epoch` is therefore the *starting* epoch —
    it is what keeps a resumed run from replaying the data its first half already saw.
    From there each worker's stream runs on continuously, which is what gives each
    epoch fresh samples.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return

    dataset = worker_info.dataset
    base_seed = getattr(dataset, "seed", 42)
    epoch = getattr(dataset, "epoch", 0)
    num_workers = worker_info.num_workers
    worker_seed = base_seed + epoch * num_workers + worker_id

    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32 - 1))
    dataset.rng = np.random.default_rng(worker_seed)
    if hasattr(dataset, "generator") and hasattr(dataset.generator, "rng"):
        dataset.generator.rng = np.random.RandomState(worker_seed % (2**32 - 1))

