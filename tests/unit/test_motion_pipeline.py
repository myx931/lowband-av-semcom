"""Tests for resumable motion extraction and validation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from av_semcom.data.grid import GridSample, GridSettings, write_manifest
from av_semcom.data.preprocessing import atomic_save_npz
from av_semcom.models.motion.config import MotionSettings
from av_semcom.models.motion.perturbations import fit_motion_normalizer
from av_semcom.models.motion.pipeline import (
    extract_motion_for_manifest,
    motion_artifact_fingerprint,
    select_motion_samples,
    validate_motion_samples,
)
from av_semcom.models.motion.sequence import load_motion_sequence
from av_semcom.models.reconstruction.backend import FakeReconstructionBackend


def _settings(root: Path) -> MotionSettings:
    config = {
        "data": {
            "root": str(root),
            "raw_video_dir": "grid/raw/video",
            "raw_audio_dir": "grid/raw/audio",
            "manifest_path": "grid/manifests/subset.jsonl",
            "failure_dir": "grid/manifests/failures",
            "processed_dir": "grid/processed",
            "speakers": ["s1"],
            "max_samples": 1,
            "fps": 25,
            "split_seed": 42,
            "validation_ratio": 0.1,
            "test_ratio": 0.1,
            "resume": True,
        },
        "motion": {
            "backend": "fake",
            "backend_revision": "test-v1",
            "output_dir": "grid/processed/motion/fake",
        },
        "experiment": {
            "output_dir": str(root / "outputs"),
            "save_sample_positions": [],
        },
    }
    data_settings = GridSettings.from_config(config)
    return MotionSettings.from_config(config, data_settings)


def _sample(root: Path) -> GridSample:
    crop_path = root / "grid/processed/face_crops/s1/s1_example.npz"
    crops = np.zeros((4, 256, 256, 3), dtype=np.uint8)
    crops[1:].fill(30)
    atomic_save_npz(
        crop_path,
        crops=crops,
        valid_mask=np.ones(4, dtype=np.bool_),
    )
    return GridSample(
        sample_id="s1_example",
        speaker_id="s1",
        video_path="grid/raw/video/s1/example",
        audio_path="grid/raw/audio/s1/example.wav",
        fps=25,
        sample_rate=25000,
        frame_count=4,
        split="pilot",
        face_crop_path="grid/processed/face_crops/s1/s1_example.npz",
        status="processed",
    )


def test_motion_pipeline_is_resumable_and_validated(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    sample = _sample(tmp_path)
    write_manifest(settings.data_settings.manifest_path, [sample])

    updated, failures, normalizer = extract_motion_for_manifest(
        settings,
        [sample],
        backend=FakeReconstructionBackend(),
    )

    assert failures == []
    assert normalizer is not None
    assert updated[0].motion_path is not None
    motion_path = tmp_path / updated[0].motion_path
    modification_time = motion_path.stat().st_mtime_ns
    report = validate_motion_samples(
        updated,
        tmp_path,
        expected_backend="fake",
        expected_revision="test-v1",
    )
    assert report.valid_count == 1
    assert report.error_count == 0

    rerun, rerun_failures, _ = extract_motion_for_manifest(
        settings,
        updated,
        backend=FakeReconstructionBackend(),
    )
    assert rerun_failures == []
    assert rerun == updated
    assert motion_path.stat().st_mtime_ns == modification_time


def test_motion_sample_selection_honors_speaker_and_limit(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    first = _sample(tmp_path)
    second = replace(first, sample_id="s1_second")
    other_speaker = replace(first, sample_id="s2_example", speaker_id="s2")

    selected = select_motion_samples(
        [first, second, other_speaker],
        settings.data_settings,
    )

    assert [sample.sample_id for sample in selected] == ["s1_example"]


def test_motion_sample_selection_skips_incomplete_preprocessing(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ready = _sample(tmp_path)
    incomplete = replace(
        ready,
        sample_id="s1_incomplete",
        face_crop_path=None,
        status="discovered",
    )

    selected = select_motion_samples(
        [incomplete, ready],
        settings.data_settings,
    )

    assert [sample.sample_id for sample in selected] == ["s1_example"]


def test_reconstruction_batch_size_does_not_invalidate_motion(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    sample = _sample(tmp_path)
    changed_config = dict(settings.config)
    changed_config["motion"] = {
        **dict(settings.config["motion"]),
        "reconstruction_batch_size": 32,
    }

    assert motion_artifact_fingerprint(
        settings,
        sample,
    ) == motion_artifact_fingerprint(
        replace(
            settings,
            reconstruction_batch_size=32,
            config=changed_config,
        ),
        sample,
    )


def test_motion_stats_use_only_configured_split(tmp_path: Path) -> None:
    base_settings = _settings(tmp_path)
    data_settings = replace(
        base_settings.data_settings,
        speakers=("s1", "s2"),
        max_samples=1,
    )
    settings = replace(
        base_settings,
        data_settings=data_settings,
        stats_path=base_settings.output_root / "train_stats.json",
        stats_split="train",
        stats_scope="train_stats",
    )
    train_sample = replace(_sample(tmp_path), split="train")
    validation_crop_path = tmp_path / "grid/processed/face_crops/s2/s2_validation.npz"
    validation_crops = np.zeros((4, 256, 256, 3), dtype=np.uint8)
    validation_crops[1:].fill(200)
    atomic_save_npz(
        validation_crop_path,
        crops=validation_crops,
        valid_mask=np.ones(4, dtype=np.bool_),
    )
    validation_sample = replace(
        train_sample,
        sample_id="s2_validation",
        speaker_id="s2",
        split="validation",
        face_crop_path="grid/processed/face_crops/s2/s2_validation.npz",
    )

    updated, failures, normalizer = extract_motion_for_manifest(
        settings,
        [train_sample, validation_sample],
        backend=FakeReconstructionBackend(),
    )

    assert failures == []
    assert normalizer is not None
    assert normalizer.scope == "train_stats"
    train_motion_path = tmp_path / updated[0].motion_path
    expected = fit_motion_normalizer(
        [load_motion_sequence(train_motion_path)],
        scope="train_stats",
    )
    assert np.allclose(normalizer.mean, expected.mean)
    assert np.allclose(normalizer.std, expected.std)
