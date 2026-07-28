from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from av_semcom.analysis.communication_report import (
    CommunicationReportSettings,
    run_communication_report,
)
from av_semcom.data.preprocessing import atomic_write_json
from av_semcom.models.predictor.artifacts import file_sha256
from av_semcom.models.selection.gate import GatePolicy

pytestmark = pytest.mark.smoke


def _settings(tmp_path: Path) -> CommunicationReportSettings:
    return CommunicationReportSettings.from_config(
        {
            "communication_report": {
                "output_dir": str(tmp_path / "output"),
                "frame_rate": 25,
                "frame_count": 75,
                "reference_frame_count": 1,
                "motion_dimension": 18,
                "methods": ["dense_jscc", "raw_magnitude", "learned_scorer"],
                "digital_bitrate_defined": False,
                "include_audio_side_information_cost": False,
                "include_reference_face_cost": False,
            }
        }
    )


def _metric_values(l1: float) -> dict[str, float]:
    return {
        "l1_mean": l1,
        "l1_std": 0.01,
        "rmse_mean": l1 * 1.5,
        "rmse_std": 0.01,
        "velocity_l1_mean": l1 * 0.5,
        "velocity_l1_std": 0.01,
        "normalized_residual_mse_mean": l1 * 2,
    }


def _build_frozen_sources(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    e5 = tmp_path / "e5"
    gate = tmp_path / "gate"
    scorer = tmp_path / "scorer"
    ablation = tmp_path / "ablation"
    fingerprint = "e5-smoke"
    atomic_write_json(
        e5 / "run_metadata.json",
        {"experiment_fingerprint": fingerprint},
    )
    atomic_write_json(
        e5 / "evaluation_complete.json",
        {"experiment_fingerprint": fingerprint, "status": "complete"},
    )
    (e5 / "test_metrics.jsonl").write_text('{"frozen": true}\n', encoding="utf-8")
    atomic_write_json(
        e5 / "video_reconstruction/summary.json",
        {
            "failure_count": 0,
            "groups": [
                {
                    "family": "prediction_only",
                    "noise_seed": None,
                    "oracle_mouth_mae": 0.20,
                    "oracle_mouth_nme": 0.30,
                    "oracle_psnr_db": 20.0,
                    "oracle_ssim": 0.70,
                    "oracle_landmark_coverage": 1.0,
                },
                {
                    "family": "jscc_awgn",
                    "channel_uses": 1,
                    "snr_db": 0.0,
                    "noise_seed": 42,
                    "oracle_mouth_mae": 0.10,
                    "oracle_mouth_nme": 0.15,
                    "oracle_psnr_db": 25.0,
                    "oracle_ssim": 0.80,
                    "oracle_landmark_coverage": 1.0,
                },
            ],
        },
    )
    policy = GatePolicy(
        experiment_fingerprint=fingerprint,
        gate_fingerprint="gate-smoke",
        selected_model_seeds={1: 42},
        thresholds_db={1: -1.0},
        calibration_snr_db=(-0.5,),
        primary_metric="l1",
        minimum_relative_improvement=0.0,
    )
    atomic_write_json(gate / "policy.json", policy.to_dict())
    test_hash = file_sha256(e5 / "test_metrics.jsonl")
    atomic_write_json(
        gate / "complete.json",
        {
            "status": "complete",
            "source_test_metrics_sha256": test_hash,
        },
    )
    gate_row: dict[str, Any] = {
        "channel_uses": 1,
        "snr_db": 0.0,
        "prediction_l1": 1.0,
        "prediction_rmse": 1.5,
        "prediction_velocity_l1": 0.5,
        "prediction_normalized_residual_mse": 2.0,
        "gated_l1": 0.5,
        "gated_rmse": 0.75,
        "gated_velocity_l1": 0.25,
        "gated_normalized_residual_mse": 1.0,
    }
    atomic_write_json(gate / "test_summary.json", {"groups": [gate_row]})
    aggregate = []
    for method, k, l1 in (
        ("dense_jscc", 18, 0.5),
        ("raw_magnitude", 2, 0.7),
        ("learned_scorer", 2, 0.6),
    ):
        aggregate.append(
            {
                "method": method,
                "channel_uses": 1,
                "snr_db": 0.0,
                "gate_transmit": True,
                "k": k,
                "seed_count": 1,
                **_metric_values(l1),
            }
        )
    atomic_write_json(
        scorer / "evaluation_summary.json",
        {
            "status": "complete",
            "maximum_dense_metric_difference": 0.0,
            "aggregate": aggregate,
        },
    )
    atomic_write_json(
        scorer / "evaluation_complete.json",
        {
            "status": "complete",
            "source_test_metrics_sha256": test_hash,
        },
    )
    atomic_write_json(ablation / "audit_summary.json", {"frozen": True})
    atomic_write_json(ablation / "audit_complete.json", {"status": "complete"})
    return e5, gate, scorer, ablation


def test_frozen_report_builds_and_resumes_without_rewriting(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    sources = _build_frozen_sources(tmp_path)
    run_dir = tmp_path / "report"

    output, summary = run_communication_report(
        settings,
        *sources,
        run_directory=run_dir,
    )
    complete_before = (output / "complete.json").stat().st_mtime_ns
    resumed, resumed_summary = run_communication_report(
        settings,
        *sources,
        run_directory=run_dir,
        resume=True,
    )

    assert resumed == output
    assert resumed_summary == summary
    assert (output / "complete.json").stat().st_mtime_ns == complete_before
    assert summary["motion_row_count"] == 4
    assert summary["video_row_count"] == 1
    assert summary["digital_bitrate_defined"] is False
    assert summary["all_transmitted_sparse_points_dominated_by_dense_same_rate"]
    payload = json.loads((output / "accounting.json").read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 2
