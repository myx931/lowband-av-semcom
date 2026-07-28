"""Frozen E3 prediction residuals used by the E5 JSCC experiment."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from av_semcom.data.grid import GridSample
from av_semcom.data.preprocessing import atomic_save_npz
from av_semcom.models.motion.perturbations import load_motion_normalizer
from av_semcom.models.predictor.artifacts import load_checkpoint, load_prediction
from av_semcom.models.predictor.config import AudioMotionSettings
from av_semcom.models.predictor.data import (
    AudioMotionDataset,
    load_audio_normalizer,
)
from av_semcom.models.predictor.model import AudioToMotionGRU
from av_semcom.models.residual.analysis import (
    compute_prediction_residual,
    normalize_residual,
)

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ResidualExample:
    """One immutable audio-prediction residual example."""

    sample_id: str
    speaker_id: str
    split: str
    prediction: FloatArray
    target: FloatArray
    raw_residual: FloatArray
    normalized_residual: FloatArray
    valid_mask: BoolArray
    transmission_mask: BoolArray

    def __post_init__(self) -> None:
        arrays = (
            self.prediction,
            self.target,
            self.raw_residual,
            self.normalized_residual,
        )
        shape = self.target.shape
        if len(shape) != 2 or shape[1] != 18 or shape[0] < 2:
            raise ValueError("motion arrays must have shape [T,18] with T >= 2")
        if any(array.shape != shape for array in arrays):
            raise ValueError("all motion arrays must share shape [T,18]")
        if any(array.dtype != np.float32 for array in arrays):
            raise ValueError("motion arrays must use float32")
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError("motion arrays contain non-finite values")
        if self.valid_mask.dtype != np.bool_ or self.valid_mask.shape != shape[:1]:
            raise ValueError("valid_mask must be boolean with shape [T]")
        if self.transmission_mask.dtype != np.bool_ or self.transmission_mask.shape != shape[:1]:
            raise ValueError("transmission_mask must be boolean with shape [T]")
        expected = self.valid_mask.copy()
        expected[0] = False
        if not np.array_equal(self.transmission_mask, expected):
            raise ValueError("transmission_mask must exclude only invalid and reference frames")
        if not self.transmission_mask.any():
            raise ValueError("example has no residual-transmission frames")
        if not np.allclose(self.raw_residual[0], 0.0, atol=1e-7, rtol=0):
            raise ValueError("reference-frame residual must be zero")
        if np.any(self.raw_residual[~self.transmission_mask] != 0):
            raise ValueError("ineligible raw residual values must be zero")
        if np.any(self.normalized_residual[~self.transmission_mask] != 0):
            raise ValueError("ineligible normalized residual values must be zero")

    @property
    def frame_count(self) -> int:
        return int(self.target.shape[0])


@dataclass(frozen=True)
class ResidualDataAudit:
    """Identity and count audit for one prepared residual collection."""

    sample_count: int
    split_counts: dict[str, int]
    speaker_counts: dict[str, int]
    split_speakers: dict[str, tuple[str, ...]]
    selected_e3_seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResidualDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Small in-memory dataset of normalized residual sequences."""

    def __init__(self, examples: Sequence[ResidualExample]) -> None:
        self.examples = tuple(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        example = self.examples[index]
        return {
            "sample_id": example.sample_id,
            "speaker_id": example.speaker_id,
            "split": example.split,
            "residual": torch.from_numpy(example.normalized_residual),
            "mask": torch.from_numpy(example.transmission_mask),
            "valid_mask": torch.from_numpy(example.valid_mask),
        }


def select_best_e3_seed(e3_run_dir: Path) -> int:
    """Select the frozen predictor seed from validation L1 only."""

    summary = _read_json(e3_run_dir / "summary.json")
    candidates = [
        group
        for group in summary.get("groups", [])
        if group.get("method") == "audio_gru" and group.get("split") == "validation"
    ]
    if not candidates:
        raise ValueError("E3 summary contains no validation audio_gru result")
    return int(min(candidates, key=lambda group: float(group["l1"]))["seed"])


def prepare_residual_examples(
    samples: Sequence[GridSample],
    predictor_settings: AudioMotionSettings,
    e3_run_dir: Path,
    *,
    splits: Sequence[str],
    inference_batch_size: int = 64,
) -> tuple[list[ResidualExample], ResidualDataAudit]:
    """Prepare frozen residuals without fitting statistics on held-out data."""

    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive")
    requested = set(splits)
    if not requested or not requested <= {"train", "validation", "test"}:
        raise ValueError("splits must be a non-empty subset of train/validation/test")
    e3_run_dir = e3_run_dir.resolve()
    experiment = _read_json(e3_run_dir / "experiment.json")
    if experiment.get("status") != "complete":
        raise ValueError("E3 experiment must be complete")
    fingerprint = str(experiment.get("experiment_fingerprint", ""))
    if not fingerprint:
        raise ValueError("E3 experiment has no fingerprint")
    selected_seed = select_best_e3_seed(e3_run_dir)
    selected_samples = sorted(
        (sample for sample in samples if sample.split in requested),
        key=lambda sample: (sample.split, sample.sample_id),
    )
    if not selected_samples:
        raise ValueError("no samples selected for residual preparation")

    motion_normalizer = load_motion_normalizer(predictor_settings.motion_stats_path)
    if motion_normalizer.scope != "train_stats":
        raise ValueError("E5 requires frozen train_stats motion normalization")
    examples: list[ResidualExample] = []
    train_samples: list[GridSample] = []
    for sample in selected_samples:
        prediction_path = (
            e3_run_dir
            / f"seed_{selected_seed}"
            / "predictions"
            / sample.split
            / f"{sample.sample_id}.npz"
        )
        if prediction_path.is_file():
            artifact = load_prediction(prediction_path, expected_fingerprint=fingerprint)
            _validate_prediction_identity(artifact, sample, selected_seed)
            examples.append(
                _make_example(
                    sample,
                    artifact["prediction"],
                    artifact["target"],
                    artifact["valid_mask"],
                    motion_normalizer.std,
                )
            )
        elif sample.split == "train":
            train_samples.append(sample)
        else:
            raise FileNotFoundError(f"missing frozen E3 prediction: {prediction_path}")

    if train_samples:
        examples.extend(
            _infer_training_predictions(
                train_samples,
                predictor_settings,
                e3_run_dir,
                selected_seed,
                fingerprint,
                inference_batch_size,
            )
        )
    examples.sort(key=lambda item: (item.split, item.sample_id))
    audit = _audit_examples(examples, selected_seed)
    for left, left_speakers in audit.split_speakers.items():
        for right, right_speakers in audit.split_speakers.items():
            if left < right and set(left_speakers) & set(right_speakers):
                raise ValueError(f"speaker leakage between {left} and {right}")
    return examples, audit


def save_residual_example(
    path: Path,
    example: ResidualExample,
    *,
    experiment_fingerprint: str,
) -> None:
    """Atomically cache one derived residual example."""

    atomic_save_npz(
        path,
        sample_id=np.asarray(example.sample_id),
        speaker_id=np.asarray(example.speaker_id),
        split=np.asarray(example.split),
        prediction=example.prediction,
        target=example.target,
        raw_residual=example.raw_residual,
        normalized_residual=example.normalized_residual,
        valid_mask=example.valid_mask,
        transmission_mask=example.transmission_mask,
        experiment_fingerprint=np.asarray(experiment_fingerprint),
    )


def load_residual_example(
    path: Path,
    *,
    expected_fingerprint: str,
) -> ResidualExample:
    """Load a cached example only when its provenance hash matches."""

    with np.load(path, allow_pickle=False) as payload:
        fingerprint = str(payload["experiment_fingerprint"].item())
        if fingerprint != expected_fingerprint:
            raise ValueError(f"residual cache fingerprint mismatch: {path}")
        return ResidualExample(
            sample_id=str(payload["sample_id"].item()),
            speaker_id=str(payload["speaker_id"].item()),
            split=str(payload["split"].item()),
            prediction=payload["prediction"].astype(np.float32),
            target=payload["target"].astype(np.float32),
            raw_residual=payload["raw_residual"].astype(np.float32),
            normalized_residual=payload["normalized_residual"].astype(np.float32),
            valid_mask=payload["valid_mask"].astype(np.bool_),
            transmission_mask=payload["transmission_mask"].astype(np.bool_),
        )


@torch.no_grad()
def _infer_training_predictions(
    samples: Sequence[GridSample],
    settings: AudioMotionSettings,
    e3_run_dir: Path,
    selected_seed: int,
    fingerprint: str,
    batch_size: int,
) -> list[ResidualExample]:
    audio_normalizer = load_audio_normalizer(e3_run_dir / "audio_stats.json")
    motion_normalizer = load_motion_normalizer(settings.motion_stats_path)
    checkpoint = load_checkpoint(
        e3_run_dir / f"seed_{selected_seed}" / "best.pt",
        expected_fingerprint=fingerprint,
    )
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("E3 checkpoint has no model_config")
    model = AudioToMotionGRU(**model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    dataset = AudioMotionDataset(
        samples,
        settings.data_settings.data_root,
        audio_normalizer,
        motion_normalizer,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    by_id = {sample.sample_id: sample for sample in samples}
    examples: list[ResidualExample] = []
    for batch in loader:
        prediction_norm = model(batch["audio"]).cpu().numpy().astype(np.float32)
        target_norm = batch["target"].cpu().numpy().astype(np.float32)
        masks = batch["mask"].cpu().numpy().astype(np.bool_)
        sample_ids = [str(value) for value in batch["sample_id"]]
        for index, sample_id in enumerate(sample_ids):
            prediction = motion_normalizer.denormalize(prediction_norm[index])
            prediction[0] = 0
            target = motion_normalizer.denormalize(target_norm[index])
            target[0] = 0
            examples.append(
                _make_example(
                    by_id[sample_id],
                    prediction,
                    target,
                    masks[index],
                    motion_normalizer.std,
                )
            )
    return examples


def _make_example(
    sample: GridSample,
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    motion_std: np.ndarray,
) -> ResidualExample:
    prediction_value = np.asarray(prediction, dtype=np.float32).copy()
    target_value = np.asarray(target, dtype=np.float32).copy()
    mask = np.asarray(valid_mask, dtype=np.bool_).copy()
    prediction_value[0] = 0
    target_value[0] = 0
    residual = compute_prediction_residual(target_value, prediction_value, mask)
    normalized = normalize_residual(residual, motion_std)
    return ResidualExample(
        sample_id=sample.sample_id,
        speaker_id=sample.speaker_id,
        split=sample.split,
        prediction=prediction_value,
        target=target_value,
        raw_residual=residual.values,
        normalized_residual=normalized.values,
        valid_mask=mask,
        transmission_mask=residual.transmission_mask,
    )


def _validate_prediction_identity(
    artifact: dict[str, Any],
    sample: GridSample,
    selected_seed: int,
) -> None:
    if artifact["sample_id"] != sample.sample_id:
        raise ValueError("E3 prediction sample_id mismatch")
    if artifact["speaker_id"] != sample.speaker_id or artifact["split"] != sample.split:
        raise ValueError("E3 prediction split or speaker mismatch")
    if artifact["method"] != "audio_gru" or artifact["seed"] != selected_seed:
        raise ValueError("E3 prediction does not belong to the validation-selected seed")


def _audit_examples(
    examples: Sequence[ResidualExample],
    selected_seed: int,
) -> ResidualDataAudit:
    split_speakers: dict[str, set[str]] = {}
    for example in examples:
        split_speakers.setdefault(example.split, set()).add(example.speaker_id)
    return ResidualDataAudit(
        sample_count=len(examples),
        split_counts=dict(Counter(example.split for example in examples)),
        speaker_counts=dict(Counter(example.speaker_id for example in examples)),
        split_speakers={
            split: tuple(sorted(speakers)) for split, speakers in split_speakers.items()
        },
        selected_e3_seed=selected_seed,
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    return payload
