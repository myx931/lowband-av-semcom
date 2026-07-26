"""Resumable GRID motion extraction and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from av_semcom.data.grid import (
    GridSample,
    GridSettings,
    read_manifest,
    relative_to_data_root,
    resolve_record_path,
    write_manifest,
)
from av_semcom.data.preprocessing import (
    FailureRecord,
    atomic_write_json,
    config_fingerprint,
    should_process,
    write_artifact_metadata,
    write_failures,
)
from av_semcom.models.motion.config import MotionSettings
from av_semcom.models.motion.perturbations import (
    MotionNormalizer,
    fit_motion_normalizer,
    load_motion_normalizer,
    save_motion_normalizer,
)
from av_semcom.models.motion.sequence import (
    MotionSequence,
    load_motion_sequence,
    save_motion_sequence,
)
from av_semcom.models.reconstruction.backend import (
    FakeReconstructionBackend,
    LivePortraitBackend,
    LivePortraitBackendConfig,
    ReconstructionBackend,
)


def create_reconstruction_backend(settings: MotionSettings) -> ReconstructionBackend:
    """Instantiate the configured backend with explicit dependency errors."""

    if settings.backend == "fake":
        backend = FakeReconstructionBackend()
        if settings.backend_revision != backend.revision:
            raise ValueError(
                f"fake backend revision must be {backend.revision}, got {settings.backend_revision}"
            )
        return backend
    if settings.model_root is None:
        raise RuntimeError(
            "MODEL_ROOT is not set. Point it to the directory that contains the "
            "manually downloaded LivePortrait weights."
        )
    return LivePortraitBackend(
        LivePortraitBackendConfig(
            repository=settings.repository,
            model_root=settings.model_root,
            expected_revision=settings.backend_revision,
            device=settings.device,
            half_precision=settings.half_precision,
            stitching=settings.stitching,
            reconstruction_batch_size=settings.reconstruction_batch_size,
        )
    )


def motion_artifact_fingerprint(
    settings: MotionSettings,
    sample: GridSample,
) -> str:
    """Return the extraction fingerprint for one sample."""

    motion_config = dict(settings.config.get("motion", {}))
    motion_config.pop("reconstruction_batch_size", None)
    motion_config.pop("stats_filename", None)
    motion_config.pop("stats_scope", None)
    motion_config.pop("stats_split", None)
    return config_fingerprint(
        {
            "stage": "motion_extraction",
            "motion": motion_config,
            "sample": {
                "sample_id": sample.sample_id,
                "face_crop_path": sample.face_crop_path,
                "frame_count": sample.frame_count,
                "fps": sample.fps,
            },
        }
    )


def select_motion_samples(
    samples: list[GridSample],
    settings: GridSettings,
) -> list[GridSample]:
    """Select preprocessing-complete samples under speaker and count limits."""

    allowed_speakers = set(settings.speakers)
    selected: list[GridSample] = []
    counts: dict[str, int] = {}
    for sample in samples:
        if sample.speaker_id not in allowed_speakers:
            continue
        if sample.status != "processed" or sample.face_crop_path is None:
            continue
        count = counts.get(sample.speaker_id, 0)
        if settings.max_samples is not None and count >= settings.max_samples:
            continue
        selected.append(sample)
        counts[sample.speaker_id] = count + 1
    return selected


def _load_face_crops(
    sample: GridSample,
    settings: MotionSettings,
) -> tuple[np.ndarray, np.ndarray]:
    if sample.face_crop_path is None:
        raise ValueError("face_crop_path is missing; complete GRID preprocessing first")
    path = resolve_record_path(
        sample.face_crop_path,
        settings.data_settings.data_root,
    )
    with np.load(path, allow_pickle=False) as data:
        crops = data["crops"].astype(np.uint8)
        valid_mask = data["valid_mask"].astype(np.bool_)
    if crops.shape != (sample.frame_count, 256, 256, 3):
        raise ValueError(
            f"face crops must have shape {(sample.frame_count, 256, 256, 3)}, got {crops.shape}"
        )
    if valid_mask.shape != (sample.frame_count,):
        raise ValueError("face-crop valid_mask does not match frame_count")
    return crops, valid_mask


def extract_motion_for_manifest(
    settings: MotionSettings,
    samples: list[GridSample],
    *,
    backend: ReconstructionBackend | None = None,
    overwrite: bool = False,
) -> tuple[list[GridSample], list[FailureRecord], MotionNormalizer | None]:
    """Extract motion for every manifest record and fit pilot statistics."""

    owns_backend = backend is None
    active_backend = backend or create_reconstruction_backend(settings)
    selected_ids = {
        sample.sample_id for sample in select_motion_samples(samples, settings.data_settings)
    }
    updated: list[GridSample] = []
    failures: list[FailureRecord] = []
    sequence_entries: list[tuple[MotionSequence, str]] = []
    try:
        if (
            active_backend.name != settings.backend
            or active_backend.revision != settings.backend_revision
        ):
            raise ValueError("configured backend name/revision does not match the active backend")
        for sample in samples:
            if sample.sample_id not in selected_ids:
                updated.append(sample)
                continue
            output = settings.output_root / sample.speaker_id / f"{sample.sample_id}.npz"
            relative_output = relative_to_data_root(
                output,
                settings.data_settings.data_root,
            )
            fingerprint = motion_artifact_fingerprint(settings, sample)
            try:
                if should_process(
                    output,
                    fingerprint,
                    resume=settings.data_settings.resume,
                    overwrite=overwrite,
                ):
                    crops, valid_mask = _load_face_crops(sample, settings)
                    sequence = active_backend.extract_motion(
                        crops,
                        valid_mask,
                        sample_id=sample.sample_id,
                        fps=sample.fps,
                        config_fingerprint=fingerprint,
                    )
                    save_motion_sequence(output, sequence)
                    write_artifact_metadata(
                        output,
                        fingerprint,
                        extra={
                            "backend": active_backend.name,
                            "backend_revision": active_backend.revision,
                            "frame_count": sequence.frame_count,
                            "lip_shape": list(sequence.lip_delta.shape),
                            "valid_coverage": float(sequence.valid_mask.mean()),
                        },
                    )
                else:
                    sequence = load_motion_sequence(output)
                if sequence.sample_id != sample.sample_id:
                    raise ValueError("motion artifact sample_id does not match manifest")
                sequence_entries.append((sequence, sample.split))
                updated.append(sample.with_artifact("motion_path", relative_output))
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                updated.append(sample)
                failures.append(
                    FailureRecord(
                        sample_id=sample.sample_id,
                        speaker_id=sample.speaker_id,
                        stage="motion_extraction",
                        reason=str(exc),
                    )
                )
    finally:
        if owns_backend:
            active_backend.close()

    write_manifest(settings.data_settings.manifest_path, updated)
    write_failures(
        settings.data_settings.failure_dir / "motion_extraction.jsonl",
        failures,
    )
    normalizer = _write_or_load_motion_stats(
        settings,
        sequence_entries,
        overwrite=overwrite,
    )
    return updated, failures, normalizer


def _write_or_load_motion_stats(
    settings: MotionSettings,
    sequence_entries: list[tuple[MotionSequence, str]],
    *,
    overwrite: bool,
) -> MotionNormalizer | None:
    sequences = [
        sequence
        for sequence, split in sequence_entries
        if settings.stats_split is None or split == settings.stats_split
    ]
    if not sequences:
        return None
    fingerprint_payload: dict[str, Any] = {
        "stage": "pilot_motion_stats",
        "backend": settings.backend,
        "backend_revision": settings.backend_revision,
        "samples": [
            {
                "sample_id": sequence.sample_id,
                "config_fingerprint": sequence.config_fingerprint,
            }
            for sequence in sequences
        ],
    }
    if settings.stats_split is not None or settings.stats_scope != "pilot_stats":
        fingerprint_payload.update(
            {
                "stage": "motion_stats",
                "stats_scope": settings.stats_scope,
                "stats_split": settings.stats_split,
            }
        )
    fingerprint = config_fingerprint(fingerprint_payload)
    if should_process(
        settings.stats_path,
        fingerprint,
        resume=settings.data_settings.resume,
        overwrite=overwrite,
    ):
        normalizer = fit_motion_normalizer(sequences, scope=settings.stats_scope)
        save_motion_normalizer(settings.stats_path, normalizer)
        write_artifact_metadata(
            settings.stats_path,
            fingerprint,
            extra={"scope": normalizer.scope, "sample_count": len(sequences)},
        )
        return normalizer
    return load_motion_normalizer(settings.stats_path)


@dataclass(frozen=True)
class MotionValidationReport:
    """Machine-readable motion manifest validation."""

    sample_count: int
    valid_count: int
    error_count: int
    errors: tuple[str, ...]
    backend_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_motion_samples(
    samples: list[GridSample],
    data_root: Path,
    *,
    expected_backend: str | None = None,
    expected_revision: str | None = None,
) -> MotionValidationReport:
    """Validate paths, schemas, identities, and pinned backend metadata."""

    errors: list[str] = []
    valid_count = 0
    backend_counts: dict[str, int] = {}
    for sample in samples:
        if sample.motion_path is None:
            errors.append(f"{sample.sample_id}: missing motion_path")
            continue
        try:
            path = resolve_record_path(sample.motion_path, data_root)
            sequence = load_motion_sequence(path)
            if sequence.sample_id != sample.sample_id:
                raise ValueError("sample_id does not match manifest")
            if sequence.frame_count != sample.frame_count:
                raise ValueError("frame_count does not match manifest")
            if not np.isclose(sequence.fps, sample.fps):
                raise ValueError("fps does not match manifest")
            if expected_backend and sequence.backend != expected_backend:
                raise ValueError(f"expected backend {expected_backend}, got {sequence.backend}")
            if expected_revision and sequence.backend_revision != expected_revision:
                raise ValueError("backend revision does not match configuration")
            backend_counts[sequence.backend] = backend_counts.get(sequence.backend, 0) + 1
            valid_count += 1
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{sample.sample_id}: {exc}")
    return MotionValidationReport(
        sample_count=len(samples),
        valid_count=valid_count,
        error_count=len(errors),
        errors=tuple(errors),
        backend_counts=backend_counts,
    )


def validate_motion_manifest(settings: MotionSettings) -> MotionValidationReport:
    """Validate the configured manifest and save a report beside it."""

    samples = select_motion_samples(
        read_manifest(settings.data_settings.manifest_path),
        settings.data_settings,
    )
    report = validate_motion_samples(
        samples,
        settings.data_settings.data_root,
        expected_backend=settings.backend,
        expected_revision=settings.backend_revision,
    )
    report_path = settings.data_settings.manifest_path.with_name("motion_validation_report.json")
    atomic_write_json(report_path, report.to_dict())
    return report
