"""Full-motion inputs derived from the immutable E5 sample cache."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from av_semcom.models.jscc.data import ResidualExample, load_residual_example
from av_semcom.models.motion.perturbations import MotionNormalizer


@dataclass(frozen=True)
class FullMotionData:
    """Source example plus an affine-standardized full-motion transport adapter."""

    source: ResidualExample
    transport: ResidualExample

    def __post_init__(self) -> None:
        if (
            self.source.sample_id != self.transport.sample_id
            or self.source.speaker_id != self.transport.speaker_id
            or self.source.split != self.transport.split
        ):
            raise ValueError("full-motion source and transport identities differ")


def adapt_full_motion(
    source: ResidualExample,
    normalizer: MotionNormalizer,
) -> FullMotionData:
    """Represent full motion as train-centered values, not audio residuals."""

    if normalizer.scope != "train_stats":
        raise ValueError("full-motion JSCC requires train_stats normalization")
    center = np.broadcast_to(normalizer.mean, source.target.shape).astype(np.float32).copy()
    center[~source.transmission_mask] = 0
    centered = source.target - center
    centered[~source.transmission_mask] = 0
    normalized = centered / normalizer.std
    normalized[~source.transmission_mask] = 0
    transport = ResidualExample(
        sample_id=source.sample_id,
        speaker_id=source.speaker_id,
        split=source.split,
        prediction=center,
        target=source.target.copy(),
        raw_residual=centered.astype(np.float32),
        normalized_residual=normalized.astype(np.float32),
        valid_mask=source.valid_mask.copy(),
        transmission_mask=source.transmission_mask.copy(),
    )
    return FullMotionData(source=source, transport=transport)


def load_full_motion_data(
    e5_run_dir: Path,
    normalizer: MotionNormalizer,
    *,
    splits: Sequence[str],
) -> list[FullMotionData]:
    """Load exactly the frozen E5 identities requested by split."""

    wanted = tuple(dict.fromkeys(str(split) for split in splits))
    if not wanted or any(split not in {"train", "validation", "test"} for split in wanted):
        raise ValueError("splits must be train, validation, or test")
    metadata = _read_json(e5_run_dir / "run_metadata.json")
    fingerprint = str(metadata.get("experiment_fingerprint", ""))
    if not fingerprint:
        raise ValueError("source E5 run has no experiment fingerprint")
    output: list[FullMotionData] = []
    for split in wanted:
        paths = sorted((e5_run_dir / "residual_data" / split).glob("*.npz"))
        if not paths:
            raise ValueError(f"source E5 run contains no {split} examples")
        for path in paths:
            source = load_residual_example(path, expected_fingerprint=fingerprint)
            if source.split != split:
                raise ValueError(f"source E5 split mismatch: {path}")
            output.append(adapt_full_motion(source, normalizer))
    _require_identity_isolation(output)
    return output


def data_audit(examples: Sequence[FullMotionData]) -> dict[str, object]:
    """Return compact identity evidence without copying the E5 arrays."""

    split_speakers: dict[str, set[str]] = {}
    for item in examples:
        split_speakers.setdefault(item.source.split, set()).add(item.source.speaker_id)
    return {
        "sample_count": len(examples),
        "split_counts": dict(Counter(item.source.split for item in examples)),
        "speaker_counts": dict(Counter(item.source.speaker_id for item in examples)),
        "split_speakers": {split: sorted(speakers) for split, speakers in split_speakers.items()},
        "source": "immutable_e5_residual_cache_targets_and_predictions",
        "transport_representation": "train_standardized_full_18d_motion",
    }


def _require_identity_isolation(examples: Sequence[FullMotionData]) -> None:
    split_speakers: dict[str, set[str]] = {}
    for item in examples:
        split_speakers.setdefault(item.source.split, set()).add(item.source.speaker_id)
    entries = sorted(split_speakers.items())
    for index, (left, speakers) in enumerate(entries):
        for right, other in entries[index + 1 :]:
            if speakers & other:
                raise ValueError(f"speaker leakage between {left} and {right}")


def _read_json(path: Path) -> dict[str, object]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return payload
