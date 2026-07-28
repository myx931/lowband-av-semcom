from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from av_semcom.models.jscc.config import JSCCSettings
from av_semcom.models.jscc.data import ResidualExample
from av_semcom.models.selection.config import (
    ResidualScorerAblationSettings,
    ResidualScorerSettings,
)
from av_semcom.models.selection.scorer import (
    ChannelAwareResidualScorer,
    hard_top_k_mask,
    raw_position_velocity_loss,
    rule_selection_mask,
)
from av_semcom.models.selection.scorer_ablation import (
    partition_validation_examples,
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
        "residual_scorer_ablation": {
            "output_dir": "outputs/test_scorer_ablation",
            "channel_uses": [2],
            "calibration_sample_count": 2,
            "partition_salt": "unit-test",
            "variants": {
                "full": {"use_snr": True, "velocity_weight": 0.5},
                "no_snr": {"use_snr": False, "velocity_weight": 0.5},
                "no_velocity": {"use_snr": True, "velocity_weight": 0.0},
                "no_snr_no_velocity": {
                    "use_snr": False,
                    "velocity_weight": 0.0,
                },
            },
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


def test_scorer_can_remove_snr_without_changing_parameter_count() -> None:
    scorer = ChannelAwareResidualScorer(
        motion_std=torch.ones(18),
        hidden_dim=8,
        max_channel_uses=2,
        use_snr=False,
    )
    residual = torch.randn((1, 4, 18))
    mask = torch.tensor([[False, True, True, True]])

    low = scorer(residual, mask, 0.0, k=4, channel_uses=2)
    high = scorer(residual, mask, 10.0, k=4, channel_uses=2)

    assert torch.equal(low.hard_mask, high.hard_mask)
    assert torch.equal(low.scores, high.scores)
    assert sum(parameter.numel() for parameter in scorer.parameters()) == 626


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


def test_ablation_config_requires_complete_two_by_two_grid() -> None:
    config = _config()
    jscc = JSCCSettings.from_config(config)
    scorer = ResidualScorerSettings.from_config(config, jscc)

    ablation = ResidualScorerAblationSettings.from_config(config, jscc, scorer)

    assert ablation.channel_uses == (2,)
    assert {(item.use_snr, item.velocity_weight) for item in ablation.variants} == {
        (False, 0.0),
        (False, 0.5),
        (True, 0.0),
        (True, 0.5),
    }

    del config["residual_scorer_ablation"]["variants"]["no_snr"]  # type: ignore[index]
    with pytest.raises(ConfigError, match="complete 2x2"):
        ResidualScorerAblationSettings.from_config(config, jscc, scorer)


def test_validation_partition_is_deterministic_and_disjoint() -> None:
    examples = [_validation_example(index) for index in range(6)]

    calibration, audit = partition_validation_examples(
        examples,
        calibration_sample_count=3,
        salt="fixed",
    )
    repeated = partition_validation_examples(
        list(reversed(examples)),
        calibration_sample_count=3,
        salt="fixed",
    )

    assert [item.sample_id for item in calibration] == [item.sample_id for item in repeated[0]]
    assert [item.sample_id for item in audit] == [item.sample_id for item in repeated[1]]
    assert {item.sample_id for item in calibration}.isdisjoint(item.sample_id for item in audit)


def _validation_example(index: int) -> ResidualExample:
    normalized = np.ones((3, 18), dtype=np.float32) * np.float32(index + 1)
    normalized[0] = 0
    valid = np.ones(3, dtype=np.bool_)
    transmission = valid.copy()
    transmission[0] = False
    raw = normalized * np.float32(0.1)
    return ResidualExample(
        sample_id=f"s6_{index}",
        speaker_id="s6",
        split="validation",
        prediction=np.zeros_like(raw),
        target=raw.copy(),
        raw_residual=raw,
        normalized_residual=normalized,
        valid_mask=valid,
        transmission_mask=transmission,
    )
