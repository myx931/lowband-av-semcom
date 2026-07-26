"""GRID corpus discovery, typed records, and JSONL manifests."""

from __future__ import annotations

import json
import os
import tempfile
import wave
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from av_semcom.data.preprocessing import FailureRecord
from av_semcom.data.splits import assign_speaker_splits
from av_semcom.utils.config import ConfigError
from av_semcom.utils.paths import resolve_data_root


@dataclass(frozen=True)
class GridSample:
    """One GRID utterance and its optional preprocessing artifacts."""

    sample_id: str
    speaker_id: str
    video_path: str
    audio_path: str
    fps: float
    sample_rate: int
    frame_count: int
    split: str
    audio_feature_path: str | None = None
    landmark_path: str | None = None
    face_crop_path: str | None = None
    motion_path: str | None = None
    status: str = "discovered"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GridSample:
        """Build a record while rejecting missing or unknown fields."""

        field_names = {field.name for field in fields(cls)}
        unknown = set(payload) - field_names
        if unknown:
            raise ValueError(f"unknown GridSample fields: {sorted(unknown)}")
        required = {
            "sample_id",
            "speaker_id",
            "video_path",
            "audio_path",
            "fps",
            "sample_rate",
            "frame_count",
            "split",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(f"missing GridSample fields: {sorted(missing)}")
        return cls(**payload)

    def with_artifact(self, field_name: str, relative_path: str) -> GridSample:
        """Return a record updated with one known artifact path."""

        if field_name not in {
            "audio_feature_path",
            "landmark_path",
            "face_crop_path",
            "motion_path",
        }:
            raise ValueError(f"unsupported artifact field: {field_name}")
        return replace(self, **{field_name: relative_path})


@dataclass(frozen=True)
class GridSettings:
    """Resolved GRID paths and preprocessing settings."""

    data_root: Path
    raw_video_root: Path
    raw_audio_root: Path
    manifest_path: Path
    failure_dir: Path
    processed_root: Path
    speakers: tuple[str, ...]
    max_samples: int | None
    fps: float
    split_seed: int
    validation_ratio: float
    test_ratio: float
    resume: bool
    config: Mapping[str, Any]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> GridSettings:
        """Resolve a ``data`` configuration mapping against ``DATA_ROOT``."""

        data = config.get("data", config)
        if not isinstance(data, Mapping):
            raise ConfigError("data configuration must be a mapping")
        root = resolve_data_root(data.get("root"))

        def resolve_relative(key: str) -> Path:
            value = data.get(key)
            if not isinstance(value, str) or not value:
                raise ConfigError(f"data.{key} must be a non-empty relative path")
            path = Path(value)
            if path.is_absolute():
                raise ConfigError(f"data.{key} must be relative to DATA_ROOT")
            resolved = (root / path).resolve()
            if root not in resolved.parents:
                raise ConfigError(f"data.{key} escapes DATA_ROOT")
            return resolved

        speakers_raw = data.get("speakers", ["s1"])
        if not isinstance(speakers_raw, list) or not all(
            isinstance(item, str) and item for item in speakers_raw
        ):
            raise ConfigError("data.speakers must be a list of speaker IDs")
        max_samples_raw = data.get("max_samples")
        max_samples = int(max_samples_raw) if max_samples_raw is not None else None
        if max_samples is not None and max_samples <= 0:
            raise ConfigError("data.max_samples must be positive or null")

        return cls(
            data_root=root,
            raw_video_root=resolve_relative("raw_video_dir"),
            raw_audio_root=resolve_relative("raw_audio_dir"),
            manifest_path=resolve_relative("manifest_path"),
            failure_dir=resolve_relative("failure_dir"),
            processed_root=resolve_relative("processed_dir"),
            speakers=tuple(speakers_raw),
            max_samples=max_samples,
            fps=float(data.get("fps", 25)),
            split_seed=int(data.get("split_seed", 42)),
            validation_ratio=float(data.get("validation_ratio", 0.1)),
            test_ratio=float(data.get("test_ratio", 0.1)),
            resume=bool(data.get("resume", True)),
            config=data,
        )


def relative_to_data_root(path: Path, data_root: Path) -> str:
    """Return a portable POSIX path below ``data_root``."""

    resolved_root = data_root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"path must be inside DATA_ROOT: {path}") from exc


def resolve_record_path(relative_path: str, data_root: Path) -> Path:
    """Resolve a manifest path while preventing traversal outside ``DATA_ROOT``."""

    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"manifest paths must be relative: {relative_path}")
    resolved_root = data_root.resolve()
    resolved = (resolved_root / path).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"manifest path escapes DATA_ROOT: {relative_path}")
    return resolved


def _read_wav_sample_rate(path: Path) -> int:
    try:
        with wave.open(str(path), "rb") as audio:
            return audio.getframerate()
    except (OSError, wave.Error) as exc:
        raise ValueError(f"could not read WAV metadata from {path}: {exc}") from exc


def discover_grid_samples(settings: GridSettings) -> tuple[list[GridSample], list[FailureRecord]]:
    """Discover paired GRID frame directories and WAV files."""

    samples: list[GridSample] = []
    failures: list[FailureRecord] = []

    for speaker_id in settings.speakers:
        video_speaker = settings.raw_video_root / speaker_id
        audio_speaker = settings.raw_audio_root / speaker_id
        video_by_id = (
            {
                directory.name: directory
                for directory in video_speaker.iterdir()
                if directory.is_dir() and any(directory.glob("*.jpg"))
            }
            if video_speaker.is_dir()
            else {}
        )
        audio_by_id = (
            {path.stem: path for path in audio_speaker.glob("*.wav") if path.is_file()}
            if audio_speaker.is_dir()
            else {}
        )
        all_ids = sorted(video_by_id.keys() | audio_by_id.keys())
        if not all_ids:
            failures.append(
                FailureRecord(
                    sample_id="*",
                    speaker_id=speaker_id,
                    stage="discovery",
                    reason=(
                        f"no paired candidates found under {video_speaker} and {audio_speaker}"
                    ),
                )
            )
            continue

        paired_count = 0
        for utterance_id in all_ids:
            # ``max_samples`` defines a bounded pilot scan.  Once enough valid
            # pairs have been selected, files outside that window must not be
            # reported as missing counterparts (for example when the complete
            # audio archive accompanies a deliberately small video subset).
            if settings.max_samples is not None and paired_count >= settings.max_samples:
                break

            sample_id = f"{speaker_id}_{utterance_id}"
            video_path = video_by_id.get(utterance_id)
            audio_path = audio_by_id.get(utterance_id)
            if video_path is None or audio_path is None:
                missing = "video frames" if video_path is None else "audio WAV"
                failures.append(
                    FailureRecord(
                        sample_id=sample_id,
                        speaker_id=speaker_id,
                        stage="discovery",
                        reason=f"missing {missing} for utterance {utterance_id}",
                    )
                )
                continue

            frame_count = len(sorted(video_path.glob("*.jpg")))
            if frame_count == 0:
                failures.append(
                    FailureRecord(
                        sample_id=sample_id,
                        speaker_id=speaker_id,
                        stage="discovery",
                        reason="video frame directory contains no JPG frames",
                    )
                )
                continue
            try:
                sample_rate = _read_wav_sample_rate(audio_path)
            except ValueError as exc:
                failures.append(
                    FailureRecord(
                        sample_id=sample_id,
                        speaker_id=speaker_id,
                        stage="discovery",
                        reason=str(exc),
                    )
                )
                continue

            samples.append(
                GridSample(
                    sample_id=sample_id,
                    speaker_id=speaker_id,
                    video_path=relative_to_data_root(video_path, settings.data_root),
                    audio_path=relative_to_data_root(audio_path, settings.data_root),
                    fps=settings.fps,
                    sample_rate=sample_rate,
                    frame_count=frame_count,
                    split="unassigned",
                )
            )
            paired_count += 1
    if not samples:
        return samples, failures

    split_by_speaker = assign_speaker_splits(
        (sample.speaker_id for sample in samples),
        seed=settings.split_seed,
        validation_ratio=settings.validation_ratio,
        test_ratio=settings.test_ratio,
    )
    assigned = [replace(sample, split=split_by_speaker[sample.speaker_id]) for sample in samples]
    return assigned, failures


def write_manifest(path: Path, samples: Iterable[GridSample]) -> None:
    """Atomically write a GRID JSONL manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for sample in samples:
            handle.write(json.dumps(asdict(sample), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.replace(temporary, path)


def read_manifest(path: Path) -> list[GridSample]:
    """Read and validate a GRID JSONL manifest."""

    if not path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {path}")
    samples: list[GridSample] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("record must be a JSON object")
                samples.append(GridSample.from_dict(payload))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid manifest record at line {line_number}: {exc}") from exc
    return samples
