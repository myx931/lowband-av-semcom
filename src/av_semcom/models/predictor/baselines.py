"""Deterministic non-learned baselines for E3."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

BASELINE_NAMES: tuple[str, ...] = (
    "zero_motion",
    "train_mean",
    "oracle_persistence",
)


def baseline_prediction(
    name: str,
    target: NDArray[np.float32],
    train_mean: NDArray[np.float32],
    *,
    source_frame_index: int = 0,
) -> NDArray[np.float32]:
    """Create one raw-motion baseline prediction with a zero reference frame."""

    if target.ndim != 2 or target.shape[1] != 18:
        raise ValueError("target must have shape [T,18]")
    if train_mean.shape != (18,):
        raise ValueError("train_mean must have shape [18]")
    if not 0 <= source_frame_index < target.shape[0]:
        raise ValueError("source_frame_index is outside target")
    if name == "zero_motion":
        prediction = np.zeros_like(target)
    elif name == "train_mean":
        prediction = np.repeat(train_mean[None], target.shape[0], axis=0).astype(np.float32)
    elif name == "oracle_persistence":
        prediction = np.empty_like(target)
        prediction[0] = target[0]
        prediction[1:] = target[:-1]
    else:
        raise ValueError(f"unknown baseline: {name}")
    prediction[source_frame_index] = 0
    return prediction.astype(np.float32, copy=False)
