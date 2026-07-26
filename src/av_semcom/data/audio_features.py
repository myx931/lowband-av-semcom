"""Log-Mel feature extraction aligned to GRID video frames."""

from __future__ import annotations

import wave
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


class AudioDependencyError(RuntimeError):
    """Raised when the optional audio backend is unavailable."""


@dataclass(frozen=True)
class AudioFeatureInfo:
    """Timing metadata recorded beside one aligned feature artifact."""

    source_sample_rate: int
    source_duration_seconds: float
    expected_duration_seconds: float
    duration_ratio: float
    alignment_mode: str


def _load_pcm_wav(path: Path) -> tuple[torch.Tensor, int]:
    """Load an uncompressed PCM WAV without optional codec backends."""

    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            raw = handle.readframes(frame_count)
    except (OSError, wave.Error) as exc:
        raise ValueError(f"could not read PCM WAV {path}: {exc}") from exc
    if channels <= 0 or frame_count <= 0:
        raise ValueError(f"audio file is empty: {path}")

    if sample_width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
    elif sample_width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648
    else:
        raise ValueError(f"unsupported PCM sample width {sample_width} bytes in {path}")
    waveform = torch.from_numpy(samples.reshape(-1, channels).T.copy())
    return waveform, sample_rate


def extract_aligned_log_mel(
    audio_path: Path,
    *,
    frame_count: int,
    fps: float,
    target_sample_rate: int,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, AudioFeatureInfo]:
    """Extract log-Mel features on the video's absolute time grid.

    Returns:
        A ``[frame_count, mel_steps_per_video_frame, n_mels]`` float32 array
        and timing metadata for auditing.
    """

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    try:
        import torchaudio
    except (ImportError, OSError) as exc:
        raise AudioDependencyError(
            "torchaudio is required for audio features. Install requirements/base.txt."
        ) from exc

    waveform, source_sample_rate = _load_pcm_wav(audio_path)
    waveform = waveform.mean(dim=0, keepdim=True)
    source_duration = waveform.shape[-1] / source_sample_rate
    if source_sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            source_sample_rate,
            target_sample_rate,
        )

    n_mels = int(config.get("n_mels", 80))
    window_size = int(config.get("window_size", 400))
    hop_size = int(config.get("hop_size", 160))
    n_fft = int(config.get("n_fft", 512))
    steps_per_frame = int(config.get("mel_steps_per_video_frame", 4))
    if min(n_mels, window_size, hop_size, n_fft, steps_per_frame) <= 0:
        raise ValueError("audio feature dimensions must be positive")
    alignment_mode = str(config.get("alignment_mode", "timestamp"))
    if alignment_mode != "timestamp":
        raise ValueError("audio.alignment_mode must be 'timestamp'")
    expected_hop = target_sample_rate / (fps * steps_per_frame)
    if not np.isclose(expected_hop, hop_size, atol=1e-6):
        raise ValueError(
            "audio hop_size must equal sample_rate / (fps * mel_steps_per_video_frame)"
        )

    expected_duration = frame_count / fps
    duration_ratio = source_duration / expected_duration
    minimum_ratio = float(config.get("minimum_duration_ratio", 0.95))
    maximum_ratio = float(config.get("maximum_duration_ratio", 1.05))
    if not 0 < minimum_ratio <= 1 <= maximum_ratio:
        raise ValueError("audio duration ratios must satisfy 0 < minimum <= 1 <= maximum")
    if not minimum_ratio <= duration_ratio <= maximum_ratio:
        raise ValueError(
            f"audio duration {source_duration:.6f}s is incompatible with "
            f"video duration {expected_duration:.6f}s "
            f"(ratio {duration_ratio:.6f}, expected {minimum_ratio:.3f}..{maximum_ratio:.3f})"
        )

    expected_samples = round(expected_duration * target_sample_rate)
    current_samples = waveform.shape[-1]
    if current_samples < expected_samples:
        waveform = torch.nn.functional.pad(waveform, (0, expected_samples - current_samples))
    else:
        waveform = waveform[..., :expected_samples]

    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        win_length=window_size,
        hop_length=hop_size,
        n_mels=n_mels,
        center=True,
        pad_mode="constant",
        power=2.0,
    )
    mel = transform(waveform)
    mel = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)(mel)
    target_steps = frame_count * steps_per_frame
    if mel.shape[-1] < target_steps:
        raise ValueError(
            f"audio produced {mel.shape[-1]} Mel steps, expected at least {target_steps}"
        )
    aligned = mel[..., :target_steps]
    aligned = aligned.squeeze(0).transpose(0, 1)
    aligned = aligned.reshape(frame_count, steps_per_frame, n_mels)
    info = AudioFeatureInfo(
        source_sample_rate=int(source_sample_rate),
        source_duration_seconds=float(source_duration),
        expected_duration_seconds=float(expected_duration),
        duration_ratio=float(duration_ratio),
        alignment_mode=alignment_mode,
    )
    return aligned.detach().cpu().numpy().astype(np.float32), info
