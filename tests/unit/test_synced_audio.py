"""Tests for recovering the timestamp-aligned GRID MPG audio track."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from av_semcom.data.grid import GridSample, GridSettings, read_manifest, write_manifest
from av_semcom.data.pipeline import extract_audio_for_manifest
from av_semcom.data.synced_audio import prepare_synchronized_audio_manifest


class _FakeExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def extract(self, source: Path, output: Path, *, sample_rate: int) -> None:
        assert source.name == "bbaf2n.mpg"
        self.calls += 1
        output.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(np.zeros(sample_rate * 3, dtype=np.int16).tobytes())


def _settings(root: Path) -> GridSettings:
    return GridSettings.from_config(
        {
            "root": str(root),
            "raw_video_dir": "grid/raw/video",
            "raw_video_mpg_dir": "grid/raw/video_mpg",
            "raw_audio_dir": "grid/raw/audio_synced",
            "source_manifest_path": "grid/manifests/source.jsonl",
            "manifest_path": "grid/manifests/synced.jsonl",
            "failure_dir": "grid/manifests/failures_synced",
            "processed_dir": "grid/processed/synced",
            "speakers": ["s1"],
            "max_samples": 20,
            "fps": 25,
            "split_seed": 42,
            "validation_ratio": 0.1,
            "test_ratio": 0.1,
            "audio_sample_rate": 16000,
            "resume": True,
            "audio_sync": {"workers": 2, "ffmpeg_executable": "ffmpeg"},
            "audio": {
                "n_mels": 80,
                "window_size": 400,
                "hop_size": 160,
                "n_fft": 512,
                "mel_steps_per_video_frame": 4,
                "alignment_mode": "timestamp",
                "minimum_duration_ratio": 0.95,
                "maximum_duration_ratio": 1.05,
            },
        }
    )


def test_synced_audio_manifest_preserves_visual_artifacts_and_resumes(
    tmp_path: Path,
) -> None:
    source_manifest = tmp_path / "grid/manifests/source.jsonl"
    sample = GridSample(
        sample_id="s1_bbaf2n",
        speaker_id="s1",
        video_path="grid/raw/video/s1/bbaf2n",
        audio_path="grid/raw/audio/s1/bbaf2n.wav",
        fps=25,
        sample_rate=25000,
        frame_count=75,
        split="pilot",
        audio_feature_path="grid/processed/old/audio.npz",
        landmark_path="grid/processed/old/landmarks.npz",
        face_crop_path="grid/processed/old/crops.npz",
        motion_path="grid/processed/old/motion.npz",
        status="processed",
    )
    write_manifest(source_manifest, [sample])
    mpg = tmp_path / "grid/raw/video_mpg/s1/bbaf2n.mpg"
    mpg.parent.mkdir(parents=True)
    mpg.write_bytes(b"fake")
    settings = _settings(tmp_path)
    extractor = _FakeExtractor()

    samples, failures, processed = prepare_synchronized_audio_manifest(
        settings,
        extractor=extractor,
    )

    assert failures == []
    assert processed == 1
    assert extractor.calls == 1
    assert samples[0].audio_path == "grid/raw/audio_synced/s1/bbaf2n.wav"
    assert samples[0].sample_rate == 16000
    assert samples[0].audio_feature_path is None
    assert samples[0].landmark_path == sample.landmark_path
    assert samples[0].face_crop_path == sample.face_crop_path
    assert samples[0].motion_path == sample.motion_path
    assert samples[0].status == "discovered"

    feature_samples, feature_failures = extract_audio_for_manifest(settings, samples)

    assert feature_failures == []
    assert feature_samples[0].status == "processed"
    assert feature_samples[0].audio_feature_path is not None

    resumed, resume_failures, resume_processed = prepare_synchronized_audio_manifest(
        settings,
        extractor=extractor,
    )
    assert resume_failures == []
    assert resume_processed == 0
    assert resumed == read_manifest(settings.manifest_path)
    assert resumed[0].audio_feature_path == feature_samples[0].audio_feature_path
    assert extractor.calls == 1
