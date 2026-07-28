from __future__ import annotations

from typing import Any

import pytest

from av_semcom.models.jscc.config import JSCCSettings
from av_semcom.models.selection.config import ChannelGateSettings
from av_semcom.models.selection.gate import (
    GatePolicy,
    apply_frozen_gate_to_test,
    derive_gate_policy,
)
from av_semcom.utils.config import ConfigError


def _config() -> dict[str, Any]:
    return {
        "channel": {
            "backend": "native_reference",
            "complex_channel_uses": [1],
            "target_power": 1.0,
        },
        "jscc_model": {"input_dim": 18, "hidden_dim": 8},
        "jscc_training": {
            "seeds": [42, 43],
            "device": "cpu",
            "snr_min_db": 0.0,
            "snr_max_db": 10.0,
        },
        "jscc_evaluation": {
            "output_dir": "outputs/test_jscc",
            "validation_snr_db": [2.5],
            "test_snr_db": [-5.0, 0.0],
            "noise_seeds": [42, 43],
        },
        "channel_gate": {
            "output_dir": "outputs/test_gate",
            "validation_snr_db": [-4.5, -3.5, -2.5],
            "noise_seeds": [42, 43],
            "primary_metric": "l1",
            "minimum_relative_improvement": 0.0,
        },
    }


def _metric_row(
    *,
    sample_id: str,
    split: str,
    condition: str,
    l1: float,
    channel_uses: int | None = None,
    model_seed: int | None = None,
    snr_db: float | None = None,
    noise_seed: int | None = None,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "speaker_id": "s6" if split == "validation" else "s7",
        "split": split,
        "condition": condition,
        "channel_uses": channel_uses,
        "model_seed": model_seed,
        "snr_db": snr_db,
        "noise_seed": noise_seed,
        "normalized_residual_mse": l1,
        "l1": l1,
        "rmse": l1,
        "velocity_l1": l1,
    }


def _calibration_rows(values: dict[float, float]) -> list[dict[str, Any]]:
    rows = [
        _metric_row(
            sample_id=sample_id,
            split="validation",
            condition="prediction_only",
            l1=1.0,
        )
        for sample_id in ("s6_a", "s6_b")
    ]
    for snr_db, l1 in values.items():
        for noise_seed in (42, 43):
            for sample_id in ("s6_a", "s6_b"):
                rows.append(
                    _metric_row(
                        sample_id=sample_id,
                        split="validation",
                        condition="jscc_awgn",
                        l1=l1,
                        channel_uses=1,
                        model_seed=43,
                        snr_db=snr_db,
                        noise_seed=noise_seed,
                    )
                )
    return rows


def test_channel_gate_config_rejects_test_snr_overlap() -> None:
    config = _config()
    config["channel_gate"]["validation_snr_db"] = [-5.0]  # type: ignore[index]
    jscc = JSCCSettings.from_config(config)

    with pytest.raises(ConfigError, match="disjoint"):
        ChannelGateSettings.from_config(config, jscc)


def test_gate_threshold_requires_a_safe_higher_snr_suffix() -> None:
    config = _config()
    jscc = JSCCSettings.from_config(config)
    gate = ChannelGateSettings.from_config(config, jscc)
    rows = _calibration_rows({-4.5: 0.9, -3.5: 1.1, -2.5: 0.8})

    policy, summary = derive_gate_policy(
        rows,
        gate,
        jscc,
        {1: 43},
        experiment_fingerprint="e5",
        gate_fingerprint="gate",
    )

    assert policy.thresholds_db == {1: -2.5}
    assert not policy.should_transmit(1, -5.0)
    assert policy.should_transmit(1, 0.0)
    assert summary["selection_used_test_metrics"] is False


def test_gate_uses_prediction_only_when_no_safe_suffix_exists() -> None:
    config = _config()
    jscc = JSCCSettings.from_config(config)
    gate = ChannelGateSettings.from_config(config, jscc)

    policy, _ = derive_gate_policy(
        _calibration_rows({-4.5: 1.2, -3.5: 1.1, -2.5: 1.01}),
        gate,
        jscc,
        {1: 43},
        experiment_fingerprint="e5",
        gate_fingerprint="gate",
    )

    assert policy.thresholds_db == {1: None}
    assert not policy.should_transmit(1, 10.0)


def test_frozen_gate_replaces_unsafe_test_rows_without_refitting() -> None:
    config = _config()
    jscc = JSCCSettings.from_config(config)
    policy = GatePolicy(
        experiment_fingerprint="e5",
        gate_fingerprint="gate",
        selected_model_seeds={1: 43},
        thresholds_db={1: -2.5},
        calibration_snr_db=(-4.5, -3.5, -2.5),
        primary_metric="l1",
        minimum_relative_improvement=0.0,
    )
    rows = [
        _metric_row(
            sample_id=sample_id,
            split="test",
            condition="prediction_only",
            l1=1.0,
        )
        for sample_id in ("s7_a", "s7_b")
    ]
    for snr_db, l1 in ((-5.0, 2.0), (0.0, 0.5)):
        for noise_seed in (42, 43):
            for sample_id in ("s7_a", "s7_b"):
                rows.append(
                    _metric_row(
                        sample_id=sample_id,
                        split="test",
                        condition="jscc_awgn",
                        l1=l1,
                        channel_uses=1,
                        model_seed=43,
                        snr_db=snr_db,
                        noise_seed=noise_seed,
                    )
                )

    gated, summary = apply_frozen_gate_to_test(rows, policy, jscc)
    unsafe = next(group for group in summary["groups"] if group["snr_db"] == -5.0)
    safe = next(group for group in summary["groups"] if group["snr_db"] == 0.0)

    assert len(gated) == 8
    assert unsafe["decision"] == "prediction_only"
    assert unsafe["gated_l1"] == pytest.approx(1.0)
    assert unsafe["always_send_l1"] == pytest.approx(2.0)
    assert safe["decision"] == "send_jscc"
    assert safe["gated_l1"] == pytest.approx(0.5)
    assert summary["test_used_for_policy_selection"] is False


def test_gate_policy_round_trip() -> None:
    policy = GatePolicy(
        experiment_fingerprint="e5",
        gate_fingerprint="gate",
        selected_model_seeds={1: 43},
        thresholds_db={1: None},
        calibration_snr_db=(-4.5, -3.5),
        primary_metric="l1",
        minimum_relative_improvement=0.0,
    )

    assert GatePolicy.from_dict(policy.to_dict()) == policy
