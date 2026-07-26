"""Synthetic end-to-end smoke test for the GRID preprocessing pipeline."""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from av_semcom.data.grid import GridSettings
from av_semcom.data.landmarks import MOUTH_LANDMARK_INDICES, FaceDetection
from av_semcom.data.pipeline import run_grid_pipeline
from av_semcom.data.validation import validate_samples


class _AlwaysDetectBackend:
    def detect(self, rgb_image: np.ndarray) -> FaceDetection:
        del rgb_image
        landmarks = np.full((len(MOUTH_LANDMARK_INDICES), 3), 0.5, dtype=np.float32)
        return FaceDetection(
            mouth_landmarks=landmarks,
            face_box=np.asarray([0.15, 0.1, 0.85, 0.9], dtype=np.float32),
        )

    def close(self) -> None:
        return None


def _write_synthetic_sample(root: Path) -> None:
    frame_directory = root / "grid/raw/video/s1/example"
    frame_directory.mkdir(parents=True)
    for index in range(5):
        Image.new("RGB", (48, 48), color=(100 + index, 60, 40)).save(
            frame_directory / f"{index:03d}.jpg"
        )

    sample_rate = 25000
    sample_count = sample_rate // 5  # Five 25 fps frames span 0.2 seconds.
    time = np.arange(sample_count, dtype=np.float32) / sample_rate
    waveform = (0.1 * np.sin(2 * math.pi * 220 * time) * 32767).astype(np.int16)
    audio_path = root / "grid/raw/audio/s1/example.wav"
    audio_path.parent.mkdir(parents=True)
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(waveform.tobytes())


def _settings(root: Path) -> GridSettings:
    return GridSettings.from_config(
        {
            "root": str(root),
            "raw_video_dir": "grid/raw/video",
            "raw_audio_dir": "grid/raw/audio",
            "manifest_path": "grid/manifests/subset.jsonl",
            "failure_dir": "grid/manifests/failures",
            "processed_dir": "grid/processed",
            "speakers": ["s1"],
            "max_samples": 20,
            "fps": 25,
            "audio_sample_rate": 16000,
            "image_size": 32,
            "split_seed": 42,
            "validation_ratio": 0.1,
            "test_ratio": 0.1,
            "resume": True,
            "audio": {
                "n_mels": 16,
                "window_size": 400,
                "hop_size": 160,
                "n_fft": 512,
                "mel_steps_per_video_frame": 4,
            },
            "landmarks": {
                "backend": "mediapipe",
                "min_detection_coverage": 0.95,
            },
            "face_crop": {"padding": 0.2},
        }
    )


@pytest.mark.smoke
def test_synthetic_grid_pipeline_is_complete_and_resumable(tmp_path: Path) -> None:
    _write_synthetic_sample(tmp_path)
    settings = _settings(tmp_path)

    samples, failures = run_grid_pipeline(settings, backend=_AlwaysDetectBackend())

    assert all(not stage_failures for stage_failures in failures.values())
    assert len(samples) == 1
    assert samples[0].status == "processed"
    report = validate_samples(samples, tmp_path, require_processed=True)
    assert report.error_count == 0

    artifact_paths = [
        tmp_path / samples[0].audio_feature_path,
        tmp_path / samples[0].landmark_path,
        tmp_path / samples[0].face_crop_path,
    ]
    modification_times = [path.stat().st_mtime_ns for path in artifact_paths]

    rerun_samples, rerun_failures = run_grid_pipeline(
        settings,
        backend=_AlwaysDetectBackend(),
    )

    assert all(not stage_failures for stage_failures in rerun_failures.values())
    assert rerun_samples == samples
    assert [path.stat().st_mtime_ns for path in artifact_paths] == modification_times
