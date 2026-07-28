from __future__ import annotations

from pathlib import Path

import pytest

from av_semcom.analysis.communication_report import (
    CommunicationReportSettings,
    _cost_fields,
    _mark_pareto,
)
from av_semcom.utils.config import ConfigError


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "communication_report": {
            "output_dir": str(tmp_path / "report"),
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


def test_grid_clip_symbol_accounting_is_explicit(tmp_path: Path) -> None:
    settings = CommunicationReportSettings.from_config(_config(tmp_path))

    assert settings.eligible_frame_count == 74
    assert settings.clip_duration_seconds == pytest.approx(3.0)

    cost = _cost_fields(
        settings,
        channel_uses=2,
        semantic_dimension_count=4,
        transmit=True,
    )
    assert cost["complex_symbols_per_clip"] == 148
    assert cost["real_channel_degrees_of_freedom_per_clip"] == 296
    assert cost["complex_symbols_per_second_clip_average"] == pytest.approx(148 / 3)
    assert cost["semantic_values_before_jscc_per_clip"] == 296
    assert cost["semantic_keep_ratio"] == pytest.approx(4 / 18)
    assert cost["digital_bitrate_bits_per_second"] is None
    assert cost["explicit_selection_index_bits_per_clip"] is None


def test_report_rejects_unmeasured_digital_bitrate_claim(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = config["communication_report"]
    assert isinstance(report, dict)
    report["digital_bitrate_defined"] = True

    with pytest.raises(ConfigError, match="cannot define bitrate"):
        CommunicationReportSettings.from_config(config)


def test_pareto_marking_uses_lower_rate_and_lower_error() -> None:
    rows = [
        {"snr_db": 0.0, "complex_symbols_per_clip": 0, "l1": 1.0},
        {"snr_db": 0.0, "complex_symbols_per_clip": 74, "l1": 0.7},
        {"snr_db": 0.0, "complex_symbols_per_clip": 74, "l1": 0.8},
        {"snr_db": 0.0, "complex_symbols_per_clip": 148, "l1": 0.6},
    ]

    _mark_pareto(rows, quality_key="l1", output_key="pareto")

    assert [row["pareto"] for row in rows] == [True, True, False, True]
