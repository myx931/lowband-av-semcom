"""Validation reports for GRID manifests and processed artifacts."""

from __future__ import annotations

import wave
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from av_semcom.data.grid import GridSample, resolve_record_path


@dataclass(frozen=True)
class ValidationReport:
    """Machine-readable manifest validation summary."""

    sample_count: int
    speaker_count: int
    split_counts: dict[str, int]
    audio_duration_ratio_min: float | None
    audio_duration_ratio_max: float | None
    error_count: int
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report."""

        return asdict(self)


def validate_samples(
    samples: Iterable[GridSample],
    data_root: Path,
    *,
    require_processed: bool = False,
) -> ValidationReport:
    """Validate paths, counts, uniqueness, and speaker split isolation."""

    records = list(samples)
    errors: list[str] = []
    seen_ids: set[str] = set()
    split_counts: dict[str, int] = {}
    speaker_splits: dict[str, set[str]] = {}
    audio_duration_ratios: list[float] = []
    for sample in records:
        if sample.sample_id in seen_ids:
            errors.append(f"duplicate sample_id: {sample.sample_id}")
        seen_ids.add(sample.sample_id)
        split_counts[sample.split] = split_counts.get(sample.split, 0) + 1
        speaker_splits.setdefault(sample.speaker_id, set()).add(sample.split)
        if sample.frame_count <= 0:
            errors.append(f"{sample.sample_id}: frame_count must be positive")
        if sample.sample_rate <= 0 or sample.fps <= 0:
            errors.append(f"{sample.sample_id}: invalid sample_rate or fps")

        paths = {
            "video_path": sample.video_path,
            "audio_path": sample.audio_path,
        }
        if require_processed:
            paths.update(
                {
                    "audio_feature_path": sample.audio_feature_path,
                    "landmark_path": sample.landmark_path,
                    "face_crop_path": sample.face_crop_path,
                }
            )
        for field_name, relative_path in paths.items():
            if not relative_path:
                errors.append(f"{sample.sample_id}: missing {field_name}")
                continue
            try:
                resolved = resolve_record_path(relative_path, data_root)
            except ValueError as exc:
                errors.append(f"{sample.sample_id}: {exc}")
                continue
            if not resolved.exists():
                errors.append(f"{sample.sample_id}: {field_name} does not exist: {relative_path}")
            elif field_name == "audio_path":
                try:
                    with wave.open(str(resolved), "rb") as handle:
                        audio_duration = handle.getnframes() / handle.getframerate()
                    expected_duration = sample.frame_count / sample.fps
                    ratio = audio_duration / expected_duration
                    audio_duration_ratios.append(ratio)
                    if not 0.95 <= ratio <= 1.05:
                        errors.append(
                            f"{sample.sample_id}: audio/video duration ratio {ratio:.6f} "
                            "is outside 0.95..1.05"
                        )
                except (OSError, wave.Error, ZeroDivisionError) as exc:
                    errors.append(f"{sample.sample_id}: invalid audio timing metadata: {exc}")

    for speaker, splits in sorted(speaker_splits.items()):
        if len(splits) > 1:
            errors.append(f"speaker {speaker} appears in multiple splits: {sorted(splits)}")
    return ValidationReport(
        sample_count=len(records),
        speaker_count=len(speaker_splits),
        split_counts=split_counts,
        audio_duration_ratio_min=(min(audio_duration_ratios) if audio_duration_ratios else None),
        audio_duration_ratio_max=(max(audio_duration_ratios) if audio_duration_ratios else None),
        error_count=len(errors),
        errors=tuple(errors),
    )
