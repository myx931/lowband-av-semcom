"""Tests for E4 configuration and masked metrics."""

from __future__ import annotations

import numpy as np
import pytest

from av_semcom.models.residual.config import ResidualSettings
from av_semcom.models.residual.experiment import _masked_motion_metrics
from av_semcom.utils.config import ConfigError


def test_masked_velocity_does_not_bridge_invalid_frames() -> None:
    target = np.zeros((4, 18), dtype=np.float32)
    candidate = np.zeros_like(target)
    candidate[1] = 1
    candidate[3] = 10
    mask = np.array([True, True, False, True])

    metrics = _masked_motion_metrics(target, candidate, mask)

    assert metrics["l1"] == pytest.approx(11 / 3)
    assert metrics["velocity_l1"] == pytest.approx(1)


def test_residual_config_requires_both_selection_spaces(tmp_path) -> None:
    config: dict[str, object] = {
        "experiment": {"output_dir": str(tmp_path), "metric_workers": 1},
        "residual": {
            "selection_spaces": ["normalized"],
            "budgets": [0, 18],
            "random_seeds": [42],
            "reconstruction_budgets": [0, 18],
            "dimension_index_bits": 5,
        },
    }

    with pytest.raises(ConfigError, match="raw then normalized"):
        ResidualSettings.from_config(config)
