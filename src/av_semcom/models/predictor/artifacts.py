"""Atomic checkpoints and prediction artifacts with compatibility hashes."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from av_semcom.data.preprocessing import atomic_save_npz


def file_sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Atomically save a PyTorch checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    *,
    expected_fingerprint: str,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint only when its experiment fingerprint matches."""

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    if payload.get("experiment_fingerprint") != expected_fingerprint:
        raise ValueError("checkpoint experiment fingerprint does not match")
    return payload


def save_prediction(
    path: Path,
    *,
    sample_id: str,
    method: str,
    split: str,
    speaker_id: str,
    prediction: NDArray[np.float32],
    target: NDArray[np.float32],
    valid_mask: NDArray[np.bool_],
    seed: int | None,
    experiment_fingerprint: str,
) -> None:
    """Save one raw-motion prediction without pickle."""

    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 18:
        raise ValueError("prediction and target must share shape [T,18]")
    if valid_mask.shape != (target.shape[0],):
        raise ValueError("valid_mask must have shape [T]")
    atomic_save_npz(
        path,
        sample_id=np.asarray(sample_id),
        method=np.asarray(method),
        split=np.asarray(split),
        speaker_id=np.asarray(speaker_id),
        prediction=prediction.astype(np.float32),
        target=target.astype(np.float32),
        valid_mask=valid_mask.astype(np.bool_),
        seed=np.asarray(-1 if seed is None else seed, dtype=np.int64),
        experiment_fingerprint=np.asarray(experiment_fingerprint),
    )


def load_prediction(
    path: Path,
    *,
    expected_fingerprint: str,
) -> dict[str, Any]:
    """Load and validate one prediction artifact."""

    with np.load(path, allow_pickle=False) as payload:
        fingerprint = str(payload["experiment_fingerprint"].item())
        if fingerprint != expected_fingerprint:
            raise ValueError("prediction experiment fingerprint does not match")
        prediction = payload["prediction"].astype(np.float32)
        target = payload["target"].astype(np.float32)
        mask = payload["valid_mask"].astype(np.bool_)
        if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 18:
            raise ValueError("prediction artifact has invalid motion shape")
        if mask.shape != (target.shape[0],):
            raise ValueError("prediction artifact has invalid mask shape")
        return {
            "sample_id": str(payload["sample_id"].item()),
            "method": str(payload["method"].item()),
            "split": str(payload["split"].item()),
            "speaker_id": str(payload["speaker_id"].item()),
            "prediction": prediction,
            "target": target,
            "valid_mask": mask,
            "seed": int(payload["seed"].item()),
            "experiment_fingerprint": fingerprint,
        }
