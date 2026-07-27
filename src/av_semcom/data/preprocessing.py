"""Shared, resumable preprocessing primitives."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FailureRecord:
    """A sample-level preprocessing failure."""

    sample_id: str
    speaker_id: str
    stage: str
    reason: str


class StaleArtifactError(RuntimeError):
    """Raised when an output exists but was made with another configuration."""


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 fingerprint for a JSON-compatible configuration."""

    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def metadata_path(output_path: Path) -> Path:
    """Return the metadata sidecar path for an output artifact."""

    return output_path.with_name(f"{output_path.name}.meta.json")


def should_process(
    output_path: Path,
    fingerprint: str,
    *,
    resume: bool = True,
    overwrite: bool = False,
) -> bool:
    """Decide whether an artifact should be generated.

    Existing artifacts are skipped only when their sidecar fingerprint matches.
    A stale or untracked artifact is never overwritten implicitly.
    """

    sidecar = metadata_path(output_path)
    if not output_path.exists() and not sidecar.exists():
        return True
    if not resume:
        raise FileExistsError(f"Output already exists: {output_path}")
    if not output_path.exists() or not sidecar.is_file():
        if overwrite:
            return True
        raise StaleArtifactError(f"Output or metadata sidecar is incomplete: {output_path}")

    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if overwrite:
            return True
        raise StaleArtifactError(f"Invalid metadata sidecar for {output_path}: {exc}") from exc
    if metadata.get("config_fingerprint") == fingerprint:
        return False
    if overwrite:
        return True
    raise StaleArtifactError(
        f"Output was created with a different configuration: {output_path}. "
        "Use --overwrite to replace it."
    )


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a JSON mapping."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Atomically save named NumPy arrays in a compressed archive."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_artifact_metadata(
    output_path: Path,
    fingerprint: str,
    *,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Write the configuration fingerprint and optional artifact metadata."""

    payload: dict[str, Any] = {"config_fingerprint": fingerprint}
    if extra:
        payload.update(extra)
    atomic_write_json(metadata_path(output_path), payload)


def write_failures(path: Path, failures: list[FailureRecord]) -> None:
    """Atomically write failures as JSON Lines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for failure in failures:
            handle.write(json.dumps(asdict(failure), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.replace(temporary, path)


def interpolate_missing(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Linearly interpolate missing time steps and edge-fill from nearest valid values."""

    if values.shape[0] != valid_mask.shape[0]:
        raise ValueError("values and valid_mask must have the same time dimension")
    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size == 0:
        raise ValueError("cannot interpolate a sequence with no valid observations")

    time = np.arange(values.shape[0])
    flattened = values.reshape(values.shape[0], -1)
    result = flattened.copy()
    for column in range(flattened.shape[1]):
        result[:, column] = np.interp(
            time,
            valid_indices,
            flattened[valid_indices, column],
        )
    return result.reshape(values.shape).astype(values.dtype, copy=False)
