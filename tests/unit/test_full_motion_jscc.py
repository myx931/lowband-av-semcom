from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from av_semcom.models.full_motion.config import full_motion_jscc_settings
from av_semcom.models.full_motion.data import adapt_full_motion
from av_semcom.models.jscc.data import ResidualExample
from av_semcom.models.motion.perturbations import MotionNormalizer
from av_semcom.utils.config import ConfigError


def _source() -> ResidualExample:
    target = np.full((4, 18), 0.4, dtype=np.float32)
    prediction = np.full((4, 18), 0.1, dtype=np.float32)
    target[0] = 0
    prediction[0] = 0
    valid = np.ones(4, dtype=np.bool_)
    transmission = valid.copy()
    transmission[0] = False
    raw = target - prediction
    raw[0] = 0
    return ResidualExample(
        sample_id="s1_demo",
        speaker_id="s1",
        split="train",
        prediction=prediction,
        target=target,
        raw_residual=raw,
        normalized_residual=raw.copy(),
        valid_mask=valid,
        transmission_mask=transmission,
    )


def test_full_motion_adapter_keeps_audio_prediction_out_of_channel_input() -> None:
    normalizer = MotionNormalizer(
        mean=np.full(18, 0.2, dtype=np.float32),
        std=np.full(18, 0.1, dtype=np.float32),
        scope="train_stats",
    )

    data = adapt_full_motion(_source(), normalizer)

    assert np.allclose(data.source.prediction[1:], 0.1)
    assert np.allclose(data.transport.prediction[1:], 0.2)
    assert np.allclose(data.transport.raw_residual[1:], 0.2)
    assert np.allclose(data.transport.normalized_residual[1:], 2.0)
    assert np.allclose(data.transport.normalized_residual[0], 0.0)


def test_full_motion_adapter_rejects_non_train_statistics() -> None:
    normalizer = MotionNormalizer(
        mean=np.zeros(18, dtype=np.float32),
        std=np.ones(18, dtype=np.float32),
        scope="pilot_stats",
    )

    with pytest.raises(ValueError, match="train_stats"):
        adapt_full_motion(_source(), normalizer)


def test_full_motion_config_changes_output_and_representation(tmp_path: Path) -> None:
    config: dict[str, object] = {
        "channel": {
            "backend": "native_reference",
            "complex_channel_uses": [1],
            "target_power": 1.0,
        },
        "jscc_model": {"input_dim": 18, "hidden_dim": 8},
        "jscc_training": {
            "seeds": [42],
            "device": "cpu",
            "snr_min_db": 0.0,
            "snr_max_db": 5.0,
        },
        "jscc_evaluation": {
            "output_dir": "outputs/residual",
            "validation_snr_db": [2.5],
            "test_snr_db": [0.0],
            "noise_seeds": [42],
        },
        "full_motion_jscc": {
            "output_dir": str(tmp_path / "full"),
            "representation": "train_standardized_full_18d_motion",
        },
    }

    settings = full_motion_jscc_settings(config)

    assert settings.output_root == (tmp_path / "full").resolve()
    assert settings.config["jscc_model"]["name"] == "full_motion_mlp_jscc"
    assert "full_motion_jscc" not in settings.config

    config["full_motion_jscc"] = {
        "output_dir": str(tmp_path / "full"),
        "representation": "audio_residual",
    }
    with pytest.raises(ConfigError, match="full 18-D motion"):
        full_motion_jscc_settings(config)
