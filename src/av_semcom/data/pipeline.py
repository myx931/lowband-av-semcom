"""Resumable GRID subset preprocessing stages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from av_semcom.data.audio_features import extract_aligned_log_mel
from av_semcom.data.face_crop import extract_face_crops
from av_semcom.data.grid import (
    GridSample,
    GridSettings,
    discover_grid_samples,
    read_manifest,
    relative_to_data_root,
    resolve_record_path,
    write_manifest,
)
from av_semcom.data.landmarks import (
    MOUTH_LANDMARK_INDICES,
    FaceLandmarkBackend,
    MediaPipeFaceMeshBackend,
    extract_landmark_sequence,
)
from av_semcom.data.preprocessing import (
    FailureRecord,
    atomic_save_npz,
    config_fingerprint,
    should_process,
    write_artifact_metadata,
    write_failures,
)


def _stage_fingerprint(
    settings: GridSettings,
    stage: str,
    sample: GridSample | None = None,
) -> str:
    payload: dict[str, Any] = {"stage": stage, "data": settings.config}
    if sample is not None:
        payload["sample"] = {
            "sample_id": sample.sample_id,
            "video_path": sample.video_path,
            "audio_path": sample.audio_path,
            "frame_count": sample.frame_count,
        }
    return config_fingerprint(payload)


def _frame_paths(sample: GridSample, settings: GridSettings) -> list[Path]:
    frame_directory = resolve_record_path(sample.video_path, settings.data_root)
    paths = sorted(frame_directory.glob("*.jpg"))
    if len(paths) != sample.frame_count:
        raise ValueError(
            f"{sample.sample_id}: manifest records {sample.frame_count} frames, "
            f"but found {len(paths)}"
        )
    return paths


def prepare_grid_subset(
    settings: GridSettings,
    *,
    overwrite: bool = False,
) -> tuple[list[GridSample], list[FailureRecord], bool]:
    """Discover a subset, write its manifest, and return whether work was done."""

    fingerprint = _stage_fingerprint(settings, "discovery")
    if not should_process(
        settings.manifest_path,
        fingerprint,
        resume=settings.resume,
        overwrite=overwrite,
    ):
        return read_manifest(settings.manifest_path), [], False

    samples, failures = discover_grid_samples(settings)
    write_manifest(settings.manifest_path, samples)
    write_artifact_metadata(
        settings.manifest_path,
        fingerprint,
        extra={"sample_count": len(samples), "failure_count": len(failures)},
    )
    write_failures(settings.failure_dir / "discovery.jsonl", failures)
    return samples, failures, True


def extract_audio_for_manifest(
    settings: GridSettings,
    samples: list[GridSample],
    *,
    overwrite: bool = False,
) -> tuple[list[GridSample], list[FailureRecord]]:
    """Extract frame-aligned log-Mel features for all manifest records."""

    audio_config = settings.config.get("audio", {})
    if not isinstance(audio_config, Mapping):
        raise ValueError("data.audio configuration must be a mapping")
    target_sample_rate = int(settings.config.get("audio_sample_rate", 16000))
    updated: list[GridSample] = []
    failures: list[FailureRecord] = []
    for sample in samples:
        output = (
            settings.processed_root
            / "audio_features"
            / sample.speaker_id
            / f"{sample.sample_id}.npz"
        )
        relative_output = relative_to_data_root(output, settings.data_root)
        fingerprint = _stage_fingerprint(settings, "audio_features", sample)
        try:
            if should_process(
                output,
                fingerprint,
                resume=settings.resume,
                overwrite=overwrite,
            ):
                features, timing = extract_aligned_log_mel(
                    resolve_record_path(sample.audio_path, settings.data_root),
                    frame_count=sample.frame_count,
                    fps=sample.fps,
                    target_sample_rate=target_sample_rate,
                    config=audio_config,
                )
                atomic_save_npz(
                    output,
                    features=features,
                    source_sample_rate=np.asarray(
                        timing.source_sample_rate,
                        dtype=np.int64,
                    ),
                    target_sample_rate=np.asarray(target_sample_rate, dtype=np.int64),
                    source_duration_seconds=np.asarray(
                        timing.source_duration_seconds,
                        dtype=np.float64,
                    ),
                    expected_duration_seconds=np.asarray(
                        timing.expected_duration_seconds,
                        dtype=np.float64,
                    ),
                )
                write_artifact_metadata(
                    output,
                    fingerprint,
                    extra={
                        "shape": list(features.shape),
                        "alignment_mode": timing.alignment_mode,
                        "source_duration_seconds": timing.source_duration_seconds,
                        "expected_duration_seconds": timing.expected_duration_seconds,
                        "duration_ratio": timing.duration_ratio,
                    },
                )
            updated_sample = sample.with_artifact("audio_feature_path", relative_output)
            if (
                updated_sample.audio_feature_path
                and updated_sample.landmark_path
                and updated_sample.face_crop_path
            ):
                updated_sample = replace(updated_sample, status="processed")
            updated.append(updated_sample)
        except (OSError, RuntimeError, ValueError) as exc:
            updated.append(sample)
            failures.append(
                FailureRecord(
                    sample_id=sample.sample_id,
                    speaker_id=sample.speaker_id,
                    stage="audio_features",
                    reason=str(exc),
                )
            )
    write_manifest(settings.manifest_path, updated)
    write_failures(settings.failure_dir / "audio_features.jsonl", failures)
    return updated, failures


def extract_landmarks_for_manifest(
    settings: GridSettings,
    samples: list[GridSample],
    *,
    backend: FaceLandmarkBackend | None = None,
    overwrite: bool = False,
) -> tuple[list[GridSample], list[FailureRecord]]:
    """Extract mouth landmarks and face boxes for all manifest records."""

    landmark_config = settings.config.get("landmarks", {})
    if not isinstance(landmark_config, Mapping):
        raise ValueError("data.landmarks configuration must be a mapping")
    if landmark_config.get("backend", "mediapipe") != "mediapipe" and backend is None:
        raise ValueError("only the mediapipe landmark backend is currently supported")
    minimum_coverage = float(landmark_config.get("min_detection_coverage", 0.95))
    owns_backend = backend is None
    active_backend = backend or MediaPipeFaceMeshBackend()

    updated: list[GridSample] = []
    failures: list[FailureRecord] = []
    try:
        for sample in samples:
            output = (
                settings.processed_root
                / "landmarks"
                / sample.speaker_id
                / f"{sample.sample_id}.npz"
            )
            relative_output = relative_to_data_root(output, settings.data_root)
            fingerprint = _stage_fingerprint(settings, "landmarks", sample)
            try:
                if should_process(
                    output,
                    fingerprint,
                    resume=settings.resume,
                    overwrite=overwrite,
                ):
                    landmarks, face_boxes, valid_mask = extract_landmark_sequence(
                        _frame_paths(sample, settings),
                        active_backend,
                        min_detection_coverage=minimum_coverage,
                    )
                    atomic_save_npz(
                        output,
                        landmarks=landmarks,
                        face_boxes=face_boxes,
                        valid_mask=valid_mask,
                        mouth_indices=np.asarray(MOUTH_LANDMARK_INDICES, dtype=np.int64),
                    )
                    write_artifact_metadata(
                        output,
                        fingerprint,
                        extra={
                            "shape": list(landmarks.shape),
                            "detection_coverage": float(valid_mask.mean()),
                        },
                    )
                updated.append(sample.with_artifact("landmark_path", relative_output))
            except (OSError, RuntimeError, ValueError) as exc:
                updated.append(sample)
                failures.append(
                    FailureRecord(
                        sample_id=sample.sample_id,
                        speaker_id=sample.speaker_id,
                        stage="landmarks",
                        reason=str(exc),
                    )
                )
    finally:
        if owns_backend:
            active_backend.close()

    write_manifest(settings.manifest_path, updated)
    write_failures(settings.failure_dir / "landmarks.jsonl", failures)
    return updated, failures


def extract_face_crops_for_manifest(
    settings: GridSettings,
    samples: list[GridSample],
    *,
    overwrite: bool = False,
) -> tuple[list[GridSample], list[FailureRecord]]:
    """Crop faces using face boxes saved by the landmark stage."""

    crop_config = settings.config.get("face_crop", {})
    if not isinstance(crop_config, Mapping):
        raise ValueError("data.face_crop configuration must be a mapping")
    image_size = int(settings.config.get("image_size", 256))
    padding = float(crop_config.get("padding", 0.2))

    updated: list[GridSample] = []
    failures: list[FailureRecord] = []
    for sample in samples:
        output = (
            settings.processed_root / "face_crops" / sample.speaker_id / f"{sample.sample_id}.npz"
        )
        relative_output = relative_to_data_root(output, settings.data_root)
        fingerprint = _stage_fingerprint(settings, "face_crops", sample)
        try:
            if sample.landmark_path is None:
                raise ValueError("landmark_path is missing; run extract_landmarks.py first")
            landmark_path = resolve_record_path(sample.landmark_path, settings.data_root)
            if should_process(
                output,
                fingerprint,
                resume=settings.resume,
                overwrite=overwrite,
            ):
                with np.load(landmark_path, allow_pickle=False) as landmark_data:
                    face_boxes = landmark_data["face_boxes"]
                    valid_mask = landmark_data["valid_mask"]
                crops = extract_face_crops(
                    _frame_paths(sample, settings),
                    face_boxes,
                    image_size=image_size,
                    padding=padding,
                )
                atomic_save_npz(output, crops=crops, valid_mask=valid_mask)
                write_artifact_metadata(
                    output,
                    fingerprint,
                    extra={"shape": list(crops.shape)},
                )
            updated_sample = sample.with_artifact("face_crop_path", relative_output)
            if (
                updated_sample.audio_feature_path
                and updated_sample.landmark_path
                and updated_sample.face_crop_path
            ):
                updated_sample = replace(updated_sample, status="processed")
            updated.append(updated_sample)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            updated.append(sample)
            failures.append(
                FailureRecord(
                    sample_id=sample.sample_id,
                    speaker_id=sample.speaker_id,
                    stage="face_crops",
                    reason=str(exc),
                )
            )
    write_manifest(settings.manifest_path, updated)
    write_failures(settings.failure_dir / "face_crops.jsonl", failures)
    return updated, failures


def run_grid_pipeline(
    settings: GridSettings,
    *,
    backend: FaceLandmarkBackend | None = None,
    overwrite: bool = False,
) -> tuple[list[GridSample], dict[str, list[FailureRecord]]]:
    """Run discovery, audio, landmarks, and crops in dependency order."""

    samples, discovery_failures, _ = prepare_grid_subset(settings, overwrite=overwrite)
    samples, audio_failures = extract_audio_for_manifest(
        settings,
        samples,
        overwrite=overwrite,
    )
    samples, landmark_failures = extract_landmarks_for_manifest(
        settings,
        samples,
        backend=backend,
        overwrite=overwrite,
    )
    samples, crop_failures = extract_face_crops_for_manifest(
        settings,
        samples,
        overwrite=overwrite,
    )
    return samples, {
        "discovery": discovery_failures,
        "audio_features": audio_failures,
        "landmarks": landmark_failures,
        "face_crops": crop_failures,
    }
