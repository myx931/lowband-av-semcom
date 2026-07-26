"""Typed GRID audio/motion pairs for the E3 predictor baseline."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from av_semcom.data.grid import GridSample, GridSettings, resolve_record_path
from av_semcom.data.preprocessing import atomic_write_json
from av_semcom.models.motion.perturbations import MotionNormalizer
from av_semcom.models.motion.sequence import load_motion_sequence


@dataclass(frozen=True)
class AudioNormalizer:
    """Per-Mel-bin statistics fitted only on training audio."""

    mean: NDArray[np.float32]
    std: NDArray[np.float32]
    scope: str = "train_stats"

    def __post_init__(self) -> None:
        if self.mean.shape != (80,) or self.std.shape != (80,):
            raise ValueError("audio normalizer mean and std must have shape (80,)")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise ValueError("audio normalizer contains non-finite values")
        if np.any(self.std <= 0):
            raise ValueError("audio normalizer std must be positive")
        if self.scope != "train_stats":
            raise ValueError("audio normalizer scope must be train_stats")

    def normalize(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        if features.ndim != 3 or features.shape[1:] != (4, 80):
            raise ValueError("audio features must have shape [T, 4, 80]")
        return np.asarray((features - self.mean) / self.std, dtype=np.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "dimension": 80,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


@dataclass(frozen=True)
class PredictorDataAudit:
    """Counts and identity checks for one E3 manifest."""

    sample_count: int
    split_counts: dict[str, int]
    speaker_counts: dict[str, int]
    split_speakers: dict[str, tuple[str, ...]]
    shape_error_count: int
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def select_predictor_samples(
    samples: Iterable[GridSample],
    settings: GridSettings,
) -> list[GridSample]:
    """Select preprocessing-complete samples with audio and motion artifacts."""

    allowed = set(settings.speakers)
    counts: dict[str, int] = {}
    selected: list[GridSample] = []
    for sample in samples:
        if sample.speaker_id not in allowed:
            continue
        if (
            sample.status != "processed"
            or sample.audio_feature_path is None
            or sample.motion_path is None
        ):
            continue
        count = counts.get(sample.speaker_id, 0)
        if settings.max_samples is not None and count >= settings.max_samples:
            continue
        selected.append(sample)
        counts[sample.speaker_id] = count + 1
    return selected


def audit_predictor_samples(
    samples: Sequence[GridSample],
    data_root: Path,
) -> PredictorDataAudit:
    """Load every pair and reject shape, path, finite-value, or split leakage errors."""

    errors: list[str] = []
    split_speakers: dict[str, set[str]] = {}
    for sample in samples:
        split_speakers.setdefault(sample.split, set()).add(sample.speaker_id)
        try:
            audio, motion, mask = load_audio_motion_pair(sample, data_root)
            if audio.shape != (sample.frame_count, 4, 80):
                raise ValueError(f"audio shape is {audio.shape}")
            if motion.shape != (sample.frame_count, 18):
                raise ValueError(f"motion shape is {motion.shape}")
            if mask.shape != (sample.frame_count,):
                raise ValueError(f"valid mask shape is {mask.shape}")
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{sample.sample_id}: {exc}")
    split_names = sorted(split_speakers)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap = split_speakers[left] & split_speakers[right]
            if overlap:
                errors.append(f"speaker leakage between {left} and {right}: {sorted(overlap)}")
    return PredictorDataAudit(
        sample_count=len(samples),
        split_counts=dict(Counter(sample.split for sample in samples)),
        speaker_counts=dict(Counter(sample.speaker_id for sample in samples)),
        split_speakers={
            split: tuple(sorted(speakers)) for split, speakers in split_speakers.items()
        },
        shape_error_count=len(errors),
        errors=tuple(errors),
    )


def load_audio_motion_pair(
    sample: GridSample,
    data_root: Path,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
    """Load one aligned raw audio/motion pair and its validity mask."""

    if sample.audio_feature_path is None or sample.motion_path is None:
        raise ValueError("audio_feature_path and motion_path are required")
    audio_path = resolve_record_path(sample.audio_feature_path, data_root)
    with np.load(audio_path, allow_pickle=False) as payload:
        audio = payload["features"].astype(np.float32)
    sequence = load_motion_sequence(resolve_record_path(sample.motion_path, data_root))
    if sequence.sample_id != sample.sample_id:
        raise ValueError("motion sample_id does not match manifest")
    motion = sequence.lip_vector.astype(np.float32)
    mask = sequence.valid_mask.astype(np.bool_)
    if audio.shape != (sample.frame_count, 4, 80):
        raise ValueError(f"expected audio shape {(sample.frame_count, 4, 80)}, got {audio.shape}")
    if motion.shape != (sample.frame_count, 18):
        raise ValueError(f"expected motion shape {(sample.frame_count, 18)}, got {motion.shape}")
    if mask.shape != (sample.frame_count,):
        raise ValueError("motion valid_mask does not match frame_count")
    if not np.isfinite(audio).all() or not np.isfinite(motion).all():
        raise ValueError("audio or motion contains non-finite values")
    return audio, motion, mask


def fit_audio_normalizer(
    train_samples: Sequence[GridSample],
    data_root: Path,
    *,
    minimum_std: float = 1e-6,
) -> AudioNormalizer:
    """Fit 80-bin audio statistics while enforcing a train-only input set."""

    if not train_samples:
        raise ValueError("at least one training sample is required")
    if any(sample.split != "train" for sample in train_samples):
        raise ValueError("audio statistics may only be fitted from train samples")
    if minimum_std <= 0:
        raise ValueError("minimum_std must be positive")
    total = np.zeros(80, dtype=np.float64)
    total_square = np.zeros(80, dtype=np.float64)
    count = 0
    for sample in train_samples:
        audio, _, mask = load_audio_motion_pair(sample, data_root)
        valid = audio[mask].reshape(-1, 80).astype(np.float64)
        total += valid.sum(axis=0)
        total_square += np.square(valid).sum(axis=0)
        count += valid.shape[0]
    if count == 0:
        raise ValueError("training audio contains no valid frames")
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 0)
    std = np.maximum(np.sqrt(variance), minimum_std)
    return AudioNormalizer(mean=mean.astype(np.float32), std=std.astype(np.float32))


def save_audio_normalizer(path: Path, normalizer: AudioNormalizer) -> None:
    atomic_write_json(path, normalizer.to_dict())


def load_audio_normalizer(path: Path) -> AudioNormalizer:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return AudioNormalizer(
        mean=np.asarray(payload["mean"], dtype=np.float32),
        std=np.asarray(payload["std"], dtype=np.float32),
        scope=str(payload["scope"]),
    )


class AudioMotionDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Lazy fixed-length GRID audio/motion dataset."""

    def __init__(
        self,
        samples: Sequence[GridSample],
        data_root: Path,
        audio_normalizer: AudioNormalizer,
        motion_normalizer: MotionNormalizer,
    ) -> None:
        self.samples = tuple(samples)
        self.data_root = data_root
        self.audio_normalizer = audio_normalizer
        self.motion_normalizer = motion_normalizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        audio, motion, mask = load_audio_motion_pair(sample, self.data_root)
        return {
            "sample_id": sample.sample_id,
            "speaker_id": sample.speaker_id,
            "split": sample.split,
            "audio": torch.from_numpy(self.audio_normalizer.normalize(audio)),
            "target": torch.from_numpy(self.motion_normalizer.normalize(motion)),
            "mask": torch.from_numpy(mask),
        }
