"""Tests for video-aligned log-Mel features."""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

from av_semcom.data.audio_features import extract_aligned_log_mel


def test_log_mel_has_configured_video_aligned_shape(tmp_path: Path) -> None:
    sample_rate = 25000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    waveform = (0.2 * np.sin(2 * math.pi * 440 * time) * 32767).astype(np.int16)
    audio_path = tmp_path / "tone.wav"
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(waveform.tobytes())

    features, source_rate = extract_aligned_log_mel(
        audio_path,
        frame_count=25,
        target_sample_rate=16000,
        config={
            "n_mels": 80,
            "window_size": 400,
            "hop_size": 160,
            "n_fft": 512,
            "mel_steps_per_video_frame": 4,
        },
    )

    assert features.shape == (25, 4, 80)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    assert source_rate == 25000
