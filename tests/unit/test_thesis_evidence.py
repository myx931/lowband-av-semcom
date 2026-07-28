from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from av_semcom.analysis.thesis_evidence import (
    ThesisEvidenceSettings,
    paired_bootstrap_mean,
    summarize_motion_pairs,
    summarize_video_pairs,
)
from av_semcom.data.preprocessing import atomic_write_json
from av_semcom.utils.config import ConfigError


def _settings(tmp_path: Path, *, expected: int = 2) -> ThesisEvidenceSettings:
    return ThesisEvidenceSettings.from_config(
        {
            "thesis_evidence": {
                "output_dir": str(tmp_path / "evidence"),
                "bootstrap_seed": 42,
                "bootstrap_resamples": 500,
                "confidence_level": 0.95,
                "expected_test_sample_count": expected,
                "qualitative_positions": list(range(expected)),
                "figure_dpi": 100,
            }
        }
    )


def test_paired_bootstrap_is_deterministic_and_uses_positive_advantage() -> None:
    values = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    first = paired_bootstrap_mean(values, resamples=1000, confidence_level=0.95, seed=7)
    second = paired_bootstrap_mean(values, resamples=1000, confidence_level=0.95, seed=7)

    assert first == second
    assert first["point_estimate"] == pytest.approx(0.25)
    assert first["ci_lower"] > 0
    assert first["direction"] == "residual_better"


def test_settings_reject_out_of_range_qualitative_position(tmp_path: Path) -> None:
    config = {
        "thesis_evidence": {
            "output_dir": str(tmp_path),
            "bootstrap_seed": 42,
            "bootstrap_resamples": 100,
            "confidence_level": 0.95,
            "expected_test_sample_count": 2,
            "qualitative_positions": [2],
            "figure_dpi": 100,
        }
    }

    with pytest.raises(ConfigError, match="position exceeds"):
        ThesisEvidenceSettings.from_config(config)


def test_motion_bootstrap_averages_noise_within_sample(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    rows = []
    for channel_uses in (1, 2, 3, 4):
        for snr_db in (-5.0, 0.0, 5.0, 10.0):
            for sample_index, sample_id in enumerate(("s1_a", "s1_b")):
                for noise_seed, offset in ((42, -0.01), (43, 0.01)):
                    advantage = float(channel_uses + sample_index) / 100.0 + offset
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "channel_uses": channel_uses,
                            "snr_db": snr_db,
                            "noise_seed": noise_seed,
                            "residual_advantage_l1": advantage,
                            "residual_advantage_rmse": advantage * 2,
                            "residual_advantage_velocity_l1": advantage * 3,
                        }
                    )

    sample_rows, ci_rows = summarize_motion_pairs(rows, settings)

    assert len(sample_rows) == 32
    assert len(ci_rows) == 48
    target = next(
        row
        for row in sample_rows
        if row["sample_id"] == "s1_a" and row["channel_uses"] == 1 and row["snr_db"] == -5.0
    )
    assert target["noise_realization_count"] == 2
    assert target["residual_advantage_l1"] == pytest.approx(0.01)


def test_video_bootstrap_pairs_sample_identity_and_condition(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    residual_dir = tmp_path / "residual"
    full_dir = tmp_path / "full"
    for directory in (residual_dir, full_dir):
        directory.mkdir()
    for sample_index, sample_id in enumerate(("s1_a", "s1_b")):
        residual_rows = []
        full_rows = []
        for channel_uses in (1, 2, 3, 4):
            for snr_db in (-5.0, 0.0, 5.0, 10.0):
                base = 0.1 + sample_index * 0.01
                common = {
                    "sample_id": sample_id,
                    "family": "jscc_awgn",
                    "channel_uses": channel_uses,
                    "snr_db": snr_db,
                    "noise_seed": 42,
                }
                residual_rows.append(
                    {
                        **common,
                        "oracle_mouth_mae": base,
                        "oracle_mouth_nme": base / 10,
                    }
                )
                full_rows.append(
                    {
                        **common,
                        "oracle_mouth_mae": base + 0.02,
                        "oracle_mouth_nme": base / 10 + 0.002,
                    }
                )
        atomic_write_json(residual_dir / f"{sample_id}.json", {"rows": residual_rows})
        atomic_write_json(full_dir / f"{sample_id}.json", {"rows": full_rows})

    pairs, ci_rows = summarize_video_pairs(residual_dir, full_dir, settings)

    assert len(pairs) == 32
    assert len(ci_rows) == 32
    assert all(row["direction"] == "residual_better" for row in ci_rows)
