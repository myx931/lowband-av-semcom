"""Helpers for deterministic random-number initialization."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch random-number generators.

    Args:
        seed: Non-negative experiment seed.
        deterministic: Request deterministic PyTorch algorithms when available.

    Raises:
        ValueError: If ``seed`` is negative.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
