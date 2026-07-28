from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from av_semcom.data.preprocessing import atomic_write_json
from av_semcom.models.jscc.config import JSCCSettings
from av_semcom.models.selection.config import ChannelGateSettings
from av_semcom.models.selection.gate import run_channel_gate_experiment

pytestmark = pytest.mark.smoke


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
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
            "output_dir": str(tmp_path / "e5-output"),
            "validation_snr_db": [2.5],
            "test_snr_db": [-5.0, 0.0],
            "noise_seeds": [42],
        },
        "channel_gate": {
            "output_dir": str(tmp_path / "gate-output"),
            "validation_snr_db": [-0.5],
            "noise_seeds": [42],
            "primary_metric": "l1",
            "minimum_relative_improvement": 0.0,
        },
    }


def _row(
    sample_id: str,
    split: str,
    condition: str,
    l1: float,
    *,
    snr_db: float | None = None,
    channel_uses: int | None = None,
    model_seed: int | None = None,
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


def test_channel_gate_artifacts_and_resume(tmp_path: Path) -> None:
    config = _config(tmp_path)
    jscc = JSCCSettings.from_config(config)
    gate = ChannelGateSettings.from_config(config, jscc)
    e5 = tmp_path / "e5"
    fingerprint = "e5-smoke"
    atomic_write_json(
        e5 / "run_metadata.json",
        {"experiment_fingerprint": fingerprint},
    )
    for name in ("training_complete.json", "evaluation_complete.json"):
        atomic_write_json(
            e5 / name,
            {"experiment_fingerprint": fingerprint, "status": "complete"},
        )
    atomic_write_json(
        e5 / "training_summary.json",
        {
            "models": [
                {
                    "channel_uses": 1,
                    "seed": 42,
                    "best_validation_normalized_mse": 0.5,
                }
            ]
        },
    )
    atomic_write_json(
        e5 / "residual_data/train_validation_complete.json",
        {"experiment_fingerprint": fingerprint, "sample_count": 2},
    )
    checkpoint = e5 / "models/c_1/seed_42/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"smoke checkpoint provenance")

    test_rows = [_row(sample_id, "test", "prediction_only", 1.0) for sample_id in ("s7_a", "s7_b")]
    for snr_db, l1 in ((-5.0, 2.0), (0.0, 0.5)):
        test_rows.extend(
            _row(
                sample_id,
                "test",
                "jscc_awgn",
                l1,
                snr_db=snr_db,
                channel_uses=1,
                model_seed=42,
                noise_seed=42,
            )
            for sample_id in ("s7_a", "s7_b")
        )
    (e5 / "test_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in test_rows),
        encoding="utf-8",
    )
    calibration_rows = [
        _row(sample_id, "validation", "prediction_only", 1.0) for sample_id in ("s6_a", "s6_b")
    ]
    calibration_rows.extend(
        _row(
            sample_id,
            "validation",
            "jscc_awgn",
            0.8,
            snr_db=-0.5,
            channel_uses=1,
            model_seed=42,
            noise_seed=42,
        )
        for sample_id in ("s6_a", "s6_b")
    )
    run_dir = tmp_path / "gate-run"

    output, summary = run_channel_gate_experiment(
        gate,
        jscc,
        None,
        e5,
        run_directory=run_dir,
        formal=False,
        calibration_rows=calibration_rows,
    )
    complete_mtime = (run_dir / "complete.json").stat().st_mtime_ns
    resumed_output, resumed = run_channel_gate_experiment(
        gate,
        jscc,
        None,
        e5,
        run_directory=run_dir,
        resume=True,
        formal=False,
    )

    assert output == resumed_output == run_dir
    assert summary == resumed
    assert summary["result_count"] == 4
    assert summary["policy"]["thresholds_db"] == {"1": -0.5}
    assert (run_dir / "complete.json").stat().st_mtime_ns == complete_mtime
    assert (run_dir / "validation_metrics.jsonl").is_file()
    assert (run_dir / "test_metrics.jsonl").is_file()
