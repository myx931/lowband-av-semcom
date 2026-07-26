"""Tests for atomic and resumable GRID frame extraction."""

from __future__ import annotations

from pathlib import Path

from av_semcom.data.grid import GridSettings
from av_semcom.data.video_frames import extract_grid_frame_sequences


class _FakeFrameExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def extract(self, source: Path, output: Path, *, quality: int) -> None:
        assert source.suffix == ".mpg"
        assert quality == 2
        self.calls += 1
        for index in range(3):
            (output / f"{index + 1:06d}.jpg").write_bytes(b"frame")


def _settings(root: Path) -> GridSettings:
    return GridSettings.from_config(
        {
            "root": str(root),
            "raw_video_dir": "grid/raw/video",
            "raw_video_mpg_dir": "grid/raw/video_mpg",
            "raw_audio_dir": "grid/raw/audio_synced",
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
            "frame_extraction": {
                "workers": 2,
                "ffmpeg_executable": "ffmpeg",
                "jpeg_quality": 2,
                "expected_frame_count": 3,
            },
        }
    )


def test_frame_extraction_is_atomic_bounded_and_resumable(tmp_path: Path) -> None:
    mpg_root = tmp_path / "grid/raw/video_mpg/s1"
    mpg_root.mkdir(parents=True)
    (mpg_root / "a.mpg").write_bytes(b"fake")
    (mpg_root / "b.mpg").write_bytes(b"outside bounded subset")
    extractor = _FakeFrameExtractor()
    settings = _settings(tmp_path)

    sample_count, processed, failures = extract_grid_frame_sequences(
        settings,
        extractor=extractor,
    )

    output = tmp_path / "grid/raw/video/s1/a"
    assert sample_count == 1
    assert processed == 1
    assert failures == []
    assert extractor.calls == 1
    assert len(list(output.glob("*.jpg"))) == 3
    modification_times = [path.stat().st_mtime_ns for path in sorted(output.glob("*.jpg"))]

    resumed_count, resumed_processed, resumed_failures = extract_grid_frame_sequences(
        settings,
        extractor=extractor,
    )

    assert resumed_count == 1
    assert resumed_processed == 0
    assert resumed_failures == []
    assert extractor.calls == 1
    assert [path.stat().st_mtime_ns for path in sorted(output.glob("*.jpg"))] == (
        modification_times
    )
