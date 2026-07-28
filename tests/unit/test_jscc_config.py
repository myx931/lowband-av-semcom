from __future__ import annotations

import pytest

from av_semcom.models.jscc.config import JSCCSettings
from av_semcom.utils.config import ConfigError


def _config() -> dict[str, object]:
    return {
        "channel": {
            "backend": "sionna",
            "complex_channel_uses": [1, 2, 4],
            "target_power": 1.0,
        },
        "jscc_model": {"input_dim": 18, "hidden_dim": 32},
        "jscc_training": {
            "seeds": [42, 43],
            "device": "cpu",
            "snr_min_db": 0.0,
            "snr_max_db": 10.0,
        },
        "jscc_evaluation": {
            "output_dir": "outputs/test_jscc",
            "validation_snr_db": [2.5, 7.5],
            "test_snr_db": [-5.0, 0.0, 5.0, 10.0],
            "noise_seeds": [42, 43, 44],
        },
    }


def test_jscc_settings_distinguish_complex_uses_from_real_dof() -> None:
    settings = JSCCSettings.from_config(_config())

    assert settings.channel_backend == "sionna"
    assert settings.channel_uses == (1, 2, 4)
    assert settings.validation_snr_db == (2.5, 7.5)
    assert settings.test_snr_db == (-5.0, 0.0, 5.0, 10.0)


def test_jscc_settings_reject_validation_test_snr_leakage() -> None:
    config = _config()
    config["jscc_evaluation"]["test_snr_db"] = [2.5]  # type: ignore[index]

    with pytest.raises(ConfigError, match="disjoint"):
        JSCCSettings.from_config(config)


def test_reference_backend_cannot_be_reported_as_formal() -> None:
    config = _config()
    config["channel"]["backend"] = "native_reference"  # type: ignore[index]
    settings = JSCCSettings.from_config(config)

    with pytest.raises(ConfigError, match="formal"):
        settings.require_formal_backend()
