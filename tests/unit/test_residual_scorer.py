from __future__ import annotations

from typing import Any

import pytest
import torch

from av_semcom.models.jscc.config import JSCCSettings
from av_semcom.models.selection.config import ResidualScorerSettings
from av_semcom.models.selection.scorer import (
    ChannelAwareResidualScorer,
    hard_top_k_mask,
    raw_position_velocity_loss,
    rule_selection_mask,
)
from av_semcom.utils.config import ConfigError


def _config() -> dict[str, Any]:
    return {
        "channel": {
            "backend": "native_reference",
            "complex_channel_uses": [1, 2],
            "target_power": 1.0,
        },
        "jscc_model": {"input_dim": 18, "hidden_dim": 8},
        "jscc_training": {
            "seeds": [42],
            "device": "cpu",
            "snr_min_db": 0.0,
            "snr_max_db": 10.0,
        },
        "jscc_evaluation": {
            "output_dir": "outputs/test_jscc",
            "validation_snr_db": [1.5],
            "test_snr_db": [0.0, 5.0],
            "noise_seeds": [42],
        },
        "residual_scorer": {
            "output_dir": "outputs/test_scorer",
            "budgets_by_channel_use": {1: 2, 2: 4},
            "hidden_dim": 16,
            "temperature": 1.0,
            "velocity_weight": 0.5,
            "seeds": [42, 43],
            "device": "cpu",
            "validation_snr_db": [2.5, 7.5],
            "noise_seeds": [42],
            "random_seeds": [42, 43],
        },
    }


def test_residual_scorer_config_requires_k_equal_two_c() -> None:
    config = _config()
    jscc = JSCCSettings.from_config(config)
    settings = ResidualScorerSettings.from_config(config, jscc)

    assert settings.budgets == {1: 2, 2: 4}
    assert settings.validation_snr_db == (2.5, 7.5)

    config["residual_scorer"]["budgets_by_channel_use"] = {1: 1, 2: 4}
    with pytest.raises(ConfigError, match="K=2C"):
        ResidualScorerSettings.from_config(config, jscc)


def test_hard_top_k_is_exact_and_prefers_low_indices_on_ties() -> None:
    scores = torch.zeros((1, 3, 18))
    mask = torch.tensor([[False, True, True]])

    selected = hard_top_k_mask(scores, mask, 2)

    assert selected[0, 0].sum() == 0
    assert selected[0, 1].sum() == 2
    assert selected[0, 1, :2].tolist() == [1.0, 1.0]


def test_scorer_forward_has_hard_mask_and_trainable_surrogate() -> None:
    scorer = ChannelAwareResidualScorer(
        motion_std=torch.ones(18),
        hidden_dim=8,
        temperature=1.0,
        max_channel_uses=2,
    )
    residual = torch.randn((2, 4, 18))
    mask = torch.tensor([[False, True, True, True], [False, True, True, True]])

    result = scorer(residual, mask, 5.0, k=4, channel_uses=2)
    result.selected_residual.sum().backward()

    assert result.scores.shape == residual.shape
    assert torch.all(result.hard_mask.sum(dim=-1)[:, 1:] == 4)
    assert torch.all(result.hard_mask[:, 0] == 0)
    assert any(parameter.grad is not None for parameter in scorer.parameters())


def test_rule_masks_distinguish_raw_and_normalized_magnitude() -> None:
    residual = torch.zeros((1, 2, 18))
    residual[0, 1, 0] = 2.0
    residual[0, 1, 1] = 1.0
    mask = torch.tensor([[False, True]])
    std = torch.ones(18)
    std[1] = 10.0

    normalized = rule_selection_mask(
        residual,
        mask,
        k=1,
        method="normalized_magnitude",
        motion_std=std,
    )
    raw = rule_selection_mask(
        residual,
        mask,
        k=1,
        method="raw_magnitude",
        motion_std=std,
    )

    assert normalized[0, 1, 0]
    assert raw[0, 1, 1]


def test_position_velocity_loss_reports_raw_components() -> None:
    target = torch.zeros((1, 3, 18))
    decoded = torch.ones((1, 3, 18))
    valid = torch.ones((1, 3), dtype=torch.bool)

    loss = raw_position_velocity_loss(
        decoded,
        target,
        valid,
        torch.ones(18),
        velocity_weight=0.5,
    )

    assert loss.position_l1.item() == pytest.approx(1.0)
    assert loss.velocity_l1.item() == pytest.approx(0.0)
    assert loss.total.item() == pytest.approx(1.0)
