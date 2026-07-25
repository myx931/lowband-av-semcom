"""Tests for GRID discovery, manifests, and speaker splitting."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from av_semcom.data.grid import (
    GridSample,
    GridSettings,
    discover_grid_samples,
    read_manifest,
    resolve_record_path,
    write_manifest,
)
from av_semcom.data.splits import assign_speaker_splits


def _write_wav(path: Path, sample_rate: int = 25000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.zeros(sample_rate // 10, dtype=np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def _settings(data_root: Path, speakers: list[str] | None = None) -> GridSettings:
    config = {
        "root": str(data_root),
        "raw_video_dir": "grid/raw/video",
        "raw_audio_dir": "grid/raw/audio",
        "manifest_path": "grid/manifests/subset.jsonl",
        "failure_dir": "grid/manifests/failures",
        "processed_dir": "grid/processed",
        "speakers": speakers or ["s1"],
        "max_samples": 20,
        "fps": 25,
        "split_seed": 42,
        "validation_ratio": 0.1,
        "test_ratio": 0.1,
        "resume": True,
    }
    return GridSettings.from_config(config)


def test_discover_grid_sample_uses_relative_paths_and_pilot_split(tmp_path: Path) -> None:
    frame_dir = tmp_path / "grid/raw/video/s1/bbaf2n"
    frame_dir.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(frame_dir / "001.jpg")
    _write_wav(tmp_path / "grid/raw/audio/s1/bbaf2n.wav")

    samples, failures = discover_grid_samples(_settings(tmp_path))

    assert failures == []
    assert len(samples) == 1
    assert samples[0].sample_id == "s1_bbaf2n"
    assert samples[0].split == "pilot"
    assert samples[0].video_path == "grid/raw/video/s1/bbaf2n"
    assert samples[0].audio_path == "grid/raw/audio/s1/bbaf2n.wav"
    assert samples[0].sample_rate == 25000


def test_discovery_records_missing_pairs(tmp_path: Path) -> None:
    frame_dir = tmp_path / "grid/raw/video/s1/video_only"
    frame_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(frame_dir / "001.jpg")
    _write_wav(tmp_path / "grid/raw/audio/s1/audio_only.wav")

    samples, failures = discover_grid_samples(_settings(tmp_path))

    assert samples == []
    assert len(failures) == 2
    assert {failure.reason for failure in failures} == {
        "missing audio WAV for utterance video_only",
        "missing video frames for utterance audio_only",
    }


def test_manifest_round_trip(tmp_path: Path) -> None:
    sample = GridSample(
        sample_id="s1_x",
        speaker_id="s1",
        video_path="grid/raw/video/s1/x",
        audio_path="grid/raw/audio/s1/x.wav",
        fps=25,
        sample_rate=25000,
        frame_count=75,
        split="pilot",
    )
    path = tmp_path / "manifest.jsonl"

    write_manifest(path, [sample])

    assert read_manifest(path) == [sample]


def test_manifest_path_cannot_escape_data_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes DATA_ROOT"):
        resolve_record_path("../outside", tmp_path)


def test_speaker_split_is_deterministic_and_isolated() -> None:
    speakers = [f"s{index}" for index in range(1, 11)]

    first = assign_speaker_splits(speakers, seed=7)
    second = assign_speaker_splits(reversed(speakers), seed=7)

    assert first == second
    assert set(first) == set(speakers)
    assert set(first.values()) == {"train", "validation", "test"}


def test_two_speakers_cannot_claim_formal_isolation() -> None:
    with pytest.raises(ValueError, match="at least three"):
        assign_speaker_splits(["s1", "s2"], seed=42)
