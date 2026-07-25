"""Log-Mel feature extraction aligned to GRID video frames."""

from __future__ import annotations

import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional


class AudioDependencyError(RuntimeError):
    """Raised when the optional audio backend is unavailable."""


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
    target_sample_rate: int,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, int]:
    """Extract log-Mel features and interpolate them to video-frame groups.

    Returns:
        A ``[frame_count, mel_steps_per_video_frame, n_mels]`` float32 array
        and the original WAV sample rate.
    """

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    try:
        import torchaudio
    except (ImportError, OSError) as exc:
        raise AudioDependencyError(
            "torchaudio is required for audio features. Install requirements/base.txt."
        ) from exc

    waveform, source_sample_rate = _load_pcm_wav(audio_path)
    waveform = waveform.mean(dim=0, keepdim=True)
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

    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        win_length=window_size,
        hop_length=hop_size,
        n_mels=n_mels,
        center=False,
        power=2.0,
    )
    mel = transform(waveform)
    mel = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)(mel)
    if mel.shape[-1] == 0:
        raise ValueError(f"audio is too short for the configured analysis window: {audio_path}")

    target_steps = frame_count * steps_per_frame
    aligned = functional.interpolate(
        mel,
        size=target_steps,
        mode="linear",
        align_corners=False,
    )
    aligned = aligned.squeeze(0).transpose(0, 1)
    aligned = aligned.reshape(frame_count, steps_per_frame, n_mels)
    return aligned.detach().cpu().numpy().astype(np.float32), int(source_sample_rate)
