from __future__ import annotations

import pytest

from av_semcom.models.jscc.config import JSCCSettings
from av_semcom.models.jscc.experiment import _summarize_test_rows
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


def test_e6_gate_config_does_not_change_e5_fingerprint_surface() -> None:
    config = _config()
    config["channel_gate"] = {"validation_snr_db": [-0.5]}
    config["residual_scorer"] = {"budgets_by_channel_use": {1: 2}}
    config["residual_scorer_ablation"] = {"channel_uses": [3, 4]}
    config["communication_report"] = {"frame_rate": 25}

    settings = JSCCSettings.from_config(config)

    assert "channel_gate" not in settings.config
    assert "residual_scorer" not in settings.config
    assert "residual_scorer_ablation" not in settings.config
    assert "communication_report" not in settings.config


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


def test_test_summary_aggregates_model_seeds_without_hiding_them() -> None:
    config = _config()
    config["channel"]["complex_channel_uses"] = [1]  # type: ignore[index]
    config["jscc_evaluation"]["test_snr_db"] = [0.0]  # type: ignore[index]
    settings = JSCCSettings.from_config(config)
    common = {
        "sample_id": "sample",
        "speaker_id": "s7",
        "split": "test",
        "noise_seed": None,
        "normalized_residual_mse": 1.0,
        "rmse": 1.0,
        "velocity_l1": 1.0,
    }
    rows = [
        {
            **common,
            "condition": "prediction_only",
            "channel_uses": None,
            "model_seed": None,
            "snr_db": None,
            "l1": 1.0,
        },
        {
            **common,
            "condition": "full_residual_oracle",
            "channel_uses": None,
            "model_seed": None,
            "snr_db": None,
            "l1": 0.0,
        },
    ]
    for seed, l1 in ((42, 0.6), (43, 0.8)):
        rows.extend(
            [
                {
                    **common,
                    "condition": "noiseless_autoencoder",
                    "channel_uses": 1,
                    "model_seed": seed,
                    "snr_db": None,
                    "l1": l1 - 0.1,
                },
                {
                    **common,
                    "condition": "jscc_awgn",
                    "channel_uses": 1,
                    "model_seed": seed,
                    "snr_db": 0.0,
                    "l1": l1,
                },
            ]
        )

    summary = _summarize_test_rows(rows, settings)
    awgn = next(row for row in summary["seed_aggregate"] if row["condition"] == "jscc_awgn")

    assert summary["schema_version"] == 2
    assert awgn["model_seed_count"] == 2
    assert awgn["l1_mean"] == pytest.approx(0.7)
    assert awgn["l1_std"] == pytest.approx(0.1)
