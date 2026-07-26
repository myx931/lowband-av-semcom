"""Recover the synchronized audio track embedded in GRID MPG files."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Protocol

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
    config_fingerprint,
    should_process,
    write_artifact_metadata,
    write_failures,
)
from av_semcom.data.splits import assign_speaker_splits
from av_semcom.utils.config import ConfigError


class SynchronizedAudioExtractor(Protocol):
    """Extract one synchronized mono PCM WAV from an audiovisual container."""

    def extract(self, source: Path, output: Path, *, sample_rate: int) -> None:
        """Write one WAV atomically or raise a descriptive exception."""


class FfmpegSynchronizedAudioExtractor:
    """FFmpeg implementation used for the official GRID MPG containers."""

    def __init__(self, executable: str = "ffmpeg") -> None:
        resolved = shutil.which(executable)
        if resolved is None:
            raise RuntimeError(f"FFmpeg executable was not found: {executable}")
        self.executable = resolved

    def extract(self, source: Path, output: Path, *, sample_rate: int) -> None:
        if not source.is_file():
            raise FileNotFoundError(f"GRID MPG does not exist: {source}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".wav",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            command = [
                self.executable,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",
                str(temporary),
            ]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                reason = completed.stderr.strip() or f"exit code {completed.returncode}"
                raise RuntimeError(f"FFmpeg audio extraction failed: {reason}")
            _validate_pcm_wav(temporary, sample_rate)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)


def _validate_pcm_wav(path: Path, expected_sample_rate: int) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            frame_count = handle.getnframes()
    except (OSError, wave.Error) as exc:
        raise ValueError(f"invalid synchronized PCM WAV {path}: {exc}") from exc
    if sample_rate != expected_sample_rate:
        raise ValueError(
            f"synchronized WAV sample rate is {sample_rate}, expected {expected_sample_rate}"
        )
    if channels != 1 or sample_width != 2 or frame_count <= 0:
        raise ValueError("synchronized WAV must be non-empty mono 16-bit PCM")
    return frame_count / sample_rate


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_path(settings: GridSettings, key: str) -> Path:
    value = settings.config.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"data.{key} must be a non-empty relative path")
    return resolve_record_path(value, settings.data_root)


def _optional_config_path(settings: GridSettings, key: str) -> Path | None:
    value = settings.config.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ConfigError(f"data.{key} must be a non-empty relative path when set")
    return resolve_record_path(value, settings.data_root)


def _selected_source_samples(
    samples: Sequence[GridSample],
    settings: GridSettings,
) -> list[GridSample]:
    allowed = set(settings.speakers)
    counts: dict[str, int] = {}
    selected: list[GridSample] = []
    for sample in samples:
        if sample.speaker_id not in allowed:
            continue
        count = counts.get(sample.speaker_id, 0)
        if settings.max_samples is not None and count >= settings.max_samples:
            continue
        selected.append(sample)
        counts[sample.speaker_id] = count + 1
    return selected


def _discover_samples_from_frames(
    settings: GridSettings,
    mpg_root: Path,
    target_sample_rate: int,
) -> list[GridSample]:
    raw_excluded = settings.config.get("excluded_sample_ids", [])
    if not isinstance(raw_excluded, list) or not all(
        isinstance(sample_id, str) and sample_id for sample_id in raw_excluded
    ):
        raise ConfigError("data.excluded_sample_ids must be a list of non-empty strings")
    excluded_sample_ids = set(raw_excluded)
    samples: list[GridSample] = []
    for speaker_id in settings.speakers:
        frame_root = settings.raw_video_root / speaker_id
        if not frame_root.is_dir():
            continue
        selected = 0
        for frame_directory in sorted(frame_root.iterdir()):
            if not frame_directory.is_dir():
                continue
            frame_paths = sorted(frame_directory.glob("*.jpg"))
            if not frame_paths:
                continue
            if settings.max_samples is not None and selected >= settings.max_samples:
                break
            utterance_id = frame_directory.name
            if f"{speaker_id}_{utterance_id}" in excluded_sample_ids:
                continue
            if not (mpg_root / speaker_id / f"{utterance_id}.mpg").is_file():
                continue
            audio_output = settings.raw_audio_root / speaker_id / f"{utterance_id}.wav"
            samples.append(
                GridSample(
                    sample_id=f"{speaker_id}_{utterance_id}",
                    speaker_id=speaker_id,
                    video_path=relative_to_data_root(
                        frame_directory,
                        settings.data_root,
                    ),
                    audio_path=relative_to_data_root(
                        audio_output,
                        settings.data_root,
                    ),
                    fps=settings.fps,
                    sample_rate=target_sample_rate,
                    frame_count=len(frame_paths),
                    split="unassigned",
                )
            )
            selected += 1
    if not samples:
        return []
    split_by_speaker = assign_speaker_splits(
        (sample.speaker_id for sample in samples),
        seed=settings.split_seed,
        validation_ratio=settings.validation_ratio,
        test_ratio=settings.test_ratio,
    )
    return [replace(sample, split=split_by_speaker[sample.speaker_id]) for sample in samples]


def prepare_synchronized_audio_manifest(
    settings: GridSettings,
    *,
    extractor: SynchronizedAudioExtractor | None = None,
    overwrite: bool = False,
) -> tuple[list[GridSample], list[FailureRecord], int]:
    """Extract embedded MPG audio and clone visual artifacts into a new manifest."""

    source_manifest = _optional_config_path(settings, "source_manifest_path")
    mpg_root = _config_path(settings, "raw_video_mpg_dir")
    sync_config = settings.config.get("audio_sync", {})
    if not isinstance(sync_config, dict):
        raise ConfigError("data.audio_sync must be a mapping")
    target_sample_rate = int(settings.config.get("audio_sample_rate", 16000))
    workers = int(sync_config.get("workers", 4))
    if target_sample_rate <= 0 or workers <= 0:
        raise ConfigError("audio sample rate and audio_sync.workers must be positive")
    executable = str(sync_config.get("ffmpeg_executable", "ffmpeg"))
    if source_manifest is None:
        source_samples = _discover_samples_from_frames(
            settings,
            mpg_root,
            target_sample_rate,
        )
        source_identity: object = [
            {
                "sample_id": sample.sample_id,
                "video_path": sample.video_path,
                "frame_count": sample.frame_count,
                "split": sample.split,
            }
            for sample in source_samples
        ]
    else:
        source_samples = _selected_source_samples(read_manifest(source_manifest), settings)
        source_identity = {
            "path": relative_to_data_root(source_manifest, settings.data_root),
            "sha256": _file_sha256(source_manifest),
        }
    fingerprint = config_fingerprint(
        {
            "stage": "synchronized_audio_manifest",
            "source": source_identity,
            "speakers": settings.speakers,
            "max_samples": settings.max_samples,
            "sample_rate": target_sample_rate,
            "mpg_root": relative_to_data_root(mpg_root, settings.data_root),
            "audio_root": relative_to_data_root(settings.raw_audio_root, settings.data_root),
            "audio_sync": sync_config,
        }
    )
    if not should_process(
        settings.manifest_path,
        fingerprint,
        resume=settings.resume,
        overwrite=overwrite,
    ):
        return read_manifest(settings.manifest_path), [], 0

    if not source_samples:
        raise ValueError("no selected GRID frame/MPG pairs were found")
    active_extractor = extractor or FfmpegSynchronizedAudioExtractor(executable)

    def process(sample: GridSample) -> tuple[GridSample | None, FailureRecord | None, bool]:
        prefix = f"{sample.speaker_id}_"
        if not sample.sample_id.startswith(prefix):
            return (
                None,
                FailureRecord(
                    sample_id=sample.sample_id,
                    speaker_id=sample.speaker_id,
                    stage="synchronized_audio",
                    reason="sample_id does not start with speaker_id",
                ),
                False,
            )
        utterance_id = sample.sample_id[len(prefix) :]
        source = mpg_root / sample.speaker_id / f"{utterance_id}.mpg"
        output = settings.raw_audio_root / sample.speaker_id / f"{utterance_id}.wav"
        artifact_fingerprint = config_fingerprint(
            {
                "stage": "synchronized_audio",
                "manifest_fingerprint": fingerprint,
                "sample_id": sample.sample_id,
                "source": relative_to_data_root(source, settings.data_root),
                "sample_rate": target_sample_rate,
            }
        )
        try:
            processed = should_process(
                output,
                artifact_fingerprint,
                resume=settings.resume,
                overwrite=overwrite,
            )
            if processed:
                active_extractor.extract(
                    source,
                    output,
                    sample_rate=target_sample_rate,
                )
                duration = _validate_pcm_wav(output, target_sample_rate)
                write_artifact_metadata(
                    output,
                    artifact_fingerprint,
                    extra={
                        "source_mpg": relative_to_data_root(source, settings.data_root),
                        "sample_rate": target_sample_rate,
                        "duration_seconds": duration,
                    },
                )
            synchronized = replace(
                sample,
                audio_path=relative_to_data_root(output, settings.data_root),
                sample_rate=target_sample_rate,
                audio_feature_path=None,
                status="discovered",
            )
            return synchronized, None, processed
        except (OSError, RuntimeError, ValueError) as exc:
            return (
                None,
                FailureRecord(
                    sample_id=sample.sample_id,
                    speaker_id=sample.speaker_id,
                    stage="synchronized_audio",
                    reason=str(exc),
                ),
                False,
            )

    if extractor is None and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(process, source_samples))
    else:
        results = [process(sample) for sample in source_samples]

    samples = [sample for sample, _, _ in results if sample is not None]
    failures = [failure for _, failure, _ in results if failure is not None]
    processed_count = sum(processed for _, _, processed in results)
    write_manifest(settings.manifest_path, samples)
    write_artifact_metadata(
        settings.manifest_path,
        fingerprint,
        extra={
            "sample_count": len(samples),
            "failure_count": len(failures),
            "source_manifest": (
                relative_to_data_root(source_manifest, settings.data_root)
                if source_manifest is not None
                else None
            ),
        },
    )
    write_failures(settings.failure_dir / "synchronized_audio.jsonl", failures)
    return samples, failures, processed_count
