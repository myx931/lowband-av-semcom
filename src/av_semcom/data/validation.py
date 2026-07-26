"""Validation reports for GRID manifests and processed artifacts."""

from __future__ import annotations

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

    for speaker, splits in sorted(speaker_splits.items()):
        if len(splits) > 1:
            errors.append(f"speaker {speaker} appears in multiple splits: {sorted(splits)}")
    return ValidationReport(
        sample_count=len(records),
        speaker_count=len(speaker_splits),
        split_counts=split_counts,
        error_count=len(errors),
        errors=tuple(errors),
    )
