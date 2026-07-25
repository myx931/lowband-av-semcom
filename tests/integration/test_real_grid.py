"""Optional integration test for a user-provided GRID s1 subset."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from av_semcom.data.grid import GridSettings
from av_semcom.data.pipeline import prepare_grid_subset


@pytest.mark.integration
def test_real_grid_s1_can_be_discovered() -> None:
    raw_root = os.environ.get("DATA_ROOT")
    if raw_root is None:
        pytest.skip("DATA_ROOT is not set")
    data_root = Path(raw_root)
    if not (data_root / "grid/raw/video/s1").is_dir():
        pytest.skip("GRID s1 video frames are not present")
    if not (data_root / "grid/raw/audio/s1").is_dir():
        pytest.skip("GRID s1 audio is not present")

    settings = GridSettings.from_config(
        {
            "root": str(data_root),
            "raw_video_dir": "grid/raw/video",
            "raw_audio_dir": "grid/raw/audio",
            "manifest_path": "grid/manifests/integration_subset.jsonl",
            "failure_dir": "grid/manifests/failures",
            "processed_dir": "grid/processed",
            "speakers": ["s1"],
            "max_samples": 20,
            "fps": 25,
            "split_seed": 42,
            "validation_ratio": 0.1,
            "test_ratio": 0.1,
            "resume": True,
        }
    )

    samples, _, _ = prepare_grid_subset(settings)

    assert samples
    assert all(sample.split == "pilot" for sample in samples)
