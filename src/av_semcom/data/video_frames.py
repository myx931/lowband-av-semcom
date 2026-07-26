"""Atomic GRID JPG extraction from the official MPG containers."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

from av_semcom.data.grid import GridSettings, relative_to_data_root, resolve_record_path
from av_semcom.data.preprocessing import (
    FailureRecord,
    config_fingerprint,
    should_process,
    write_artifact_metadata,
    write_failures,
)
from av_semcom.utils.config import ConfigError


class GridFrameExtractor(Protocol):
    """Extract one ordered JPG sequence from a GRID MPG."""

    def extract(self, source: Path, output: Path, *, quality: int) -> None:
        """Write frames below an empty temporary output directory."""


class FfmpegGridFrameExtractor:
    """FFmpeg-backed frame extraction without audio decoding."""

    def __init__(self, executable: str = "ffmpeg") -> None:
        resolved = shutil.which(executable)
        if resolved is None:
            raise RuntimeError(f"FFmpeg executable was not found: {executable}")
        self.executable = resolved

    def extract(self, source: Path, output: Path, *, quality: int) -> None:
        if not source.is_file():
            raise FileNotFoundError(f"GRID MPG does not exist: {source}")
        completed = subprocess.run(
            [
                self.executable,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-an",
                "-q:v",
                str(quality),
                "-start_number",
                "1",
                str(output / "%06d.jpg"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            reason = completed.stderr.strip() or f"exit code {completed.returncode}"
            raise RuntimeError(f"FFmpeg frame extraction failed: {reason}")


def _config_path(settings: GridSettings, key: str) -> Path:
    value = settings.config.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"data.{key} must be a non-empty relative path")
    return resolve_record_path(value, settings.data_root)


def _replace_directory_atomically(temporary: Path, output: Path) -> None:
    backup: Path | None = None
    if output.exists():
        backup = Path(
            tempfile.mkdtemp(
                dir=output.parent,
                prefix=f".{output.name}.backup.",
            )
        )
        backup.rmdir()
        os.replace(output, backup)
    try:
        os.replace(temporary, output)
    except BaseException:
        if backup is not None and backup.exists() and not output.exists():
            os.replace(backup, output)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def extract_grid_frame_sequences(
    settings: GridSettings,
    *,
    extractor: GridFrameExtractor | None = None,
    overwrite: bool = False,
) -> tuple[int, int, list[FailureRecord]]:
    """Extract the configured bounded speaker subset into atomic frame directories."""

    mpg_root = _config_path(settings, "raw_video_mpg_dir")
    frame_config = settings.config.get("frame_extraction", {})
    if not isinstance(frame_config, dict):
        raise ConfigError("data.frame_extraction must be a mapping")
    executable = str(frame_config.get("ffmpeg_executable", "ffmpeg"))
    workers = int(frame_config.get("workers", 4))
    quality = int(frame_config.get("jpeg_quality", 2))
    expected_frame_count = int(frame_config.get("expected_frame_count", 75))
    if workers <= 0 or quality <= 0 or expected_frame_count <= 0:
        raise ConfigError("frame extraction workers, quality, and frame count must be positive")
    active_extractor = extractor or FfmpegGridFrameExtractor(executable)

    candidates: list[tuple[str, Path]] = []
    for speaker_id in settings.speakers:
        speaker_root = mpg_root / speaker_id
        paths = sorted(speaker_root.glob("*.mpg")) if speaker_root.is_dir() else []
        if settings.max_samples is not None:
            paths = paths[: settings.max_samples]
        candidates.extend((speaker_id, path) for path in paths)
    if not candidates:
        raise ValueError("no GRID MPG files matched the configured speakers")

    def process(candidate: tuple[str, Path]) -> tuple[bool, FailureRecord | None]:
        speaker_id, source = candidate
        output = settings.raw_video_root / speaker_id / source.stem
        fingerprint = config_fingerprint(
            {
                "stage": "grid_video_frames",
                "speaker_id": speaker_id,
                "source": relative_to_data_root(source, settings.data_root),
                "fps": settings.fps,
                "frame_extraction": frame_config,
            }
        )
        temporary: Path | None = None
        try:
            processed = should_process(
                output,
                fingerprint,
                resume=settings.resume,
                overwrite=overwrite,
            )
            if not processed:
                return False, None
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(
                tempfile.mkdtemp(
                    dir=output.parent,
                    prefix=f".{output.name}.frames.",
                )
            )
            active_extractor.extract(source, temporary, quality=quality)
            frame_count = len(list(temporary.glob("*.jpg")))
            if frame_count != expected_frame_count:
                raise ValueError(f"extracted {frame_count} frames, expected {expected_frame_count}")
            _replace_directory_atomically(temporary, output)
            temporary = None
            write_artifact_metadata(
                output,
                fingerprint,
                extra={
                    "source_mpg": relative_to_data_root(source, settings.data_root),
                    "frame_count": frame_count,
                    "fps": settings.fps,
                },
            )
            return True, None
        except (OSError, RuntimeError, ValueError) as exc:
            return False, FailureRecord(
                sample_id=f"{speaker_id}_{source.stem}",
                speaker_id=speaker_id,
                stage="video_frames",
                reason=str(exc),
            )
        finally:
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)

    if extractor is None and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(process, candidates))
    else:
        results = [process(candidate) for candidate in candidates]
    failures = [failure for _, failure in results if failure is not None]
    processed_count = sum(processed for processed, _ in results)
    write_failures(settings.failure_dir / "video_frames.jsonl", failures)
    return len(candidates), processed_count, failures
