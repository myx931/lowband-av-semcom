"""Tests for deterministic sensitivity perturbations."""

from __future__ import annotations

import numpy as np

from av_semcom.models.motion.perturbations import (
    PerturbationCondition,
    apply_perturbation,
)


def test_random_dropout_is_deterministic_and_keeps_fixed_count() -> None:
    values = np.ones((4, 18), dtype=np.float32)
    condition = PerturbationCondition(
        "random",
        "random_dropout",
        value=0.25,
        seed=42,
    )

    first = apply_perturbation(values, condition, source_frame_index=0)
    second = apply_perturbation(values, condition, source_frame_index=0)

    assert np.array_equal(first, second)
    assert np.count_nonzero(first[1:], axis=1).tolist() == [4, 4, 4]
    assert np.array_equal(first[0], values[0])


def test_quantization_clips_and_uses_requested_levels() -> None:
    values = np.linspace(-5, 5, 36, dtype=np.float32).reshape(2, 18)
    condition = PerturbationCondition("two_bit", "quantization", value=2)

    result = apply_perturbation(values, condition, source_frame_index=0)

    assert np.array_equal(result[0], values[0])
    assert result[1].min() >= -3
    assert result[1].max() <= 3
    assert np.unique(result[1]).size <= 4


def test_magnitude_sparsity_retains_largest_coordinates() -> None:
    values = np.arange(36, dtype=np.float32).reshape(2, 18)
    condition = PerturbationCondition("top", "magnitude_sparsity", value=0.1)

    result = apply_perturbation(values, condition, source_frame_index=0)

    assert np.count_nonzero(result[1]) == 2
    assert result[1, -2:].tolist() == values[1, -2:].tolist()
