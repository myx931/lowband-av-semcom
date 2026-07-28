"""Focused tests for pure prediction-residual analysis."""

from __future__ import annotations

import numpy as np
import pytest

from av_semcom.models.residual.analysis import (
    DIMENSION_INDEX_BITS,
    ResidualSequence,
    compute_energy_concentration,
    compute_per_dimension_metrics,
    compute_prediction_residual,
    normalize_residual,
    reconstruct_motion,
    retain_random_k,
    retain_top_k,
    selection_accounting,
)


def _residual_fixture() -> ResidualSequence:
    target = np.zeros((4, 18), dtype=np.float32)
    target[1] = np.arange(1, 19, dtype=np.float32)
    target[2] = np.arange(18, 0, -1, dtype=np.float32)
    target[3] = 100
    prediction = np.zeros_like(target)
    valid_mask = np.array([True, True, True, False])
    return compute_prediction_residual(target, prediction, valid_mask)


def test_prediction_residual_mask_first_frame_and_reconstruction() -> None:
    target = np.arange(54, dtype=np.float32).reshape(3, 18)
    target[0] = 0
    prediction = np.ones_like(target)
    prediction[0] = 0
    valid_mask = np.array([True, True, False])

    residual = compute_prediction_residual(target, prediction, valid_mask)
    reconstructed = reconstruct_motion(prediction, residual)

    assert residual.values.shape == (3, 18)
    assert residual.valid_mask.shape == (3,)
    assert np.array_equal(residual.values[0], np.zeros(18, dtype=np.float32))
    assert np.array_equal(residual.values[2], np.zeros(18, dtype=np.float32))
    assert np.array_equal(reconstructed[1], target[1])
    assert np.array_equal(reconstructed[2], prediction[2])
    with pytest.raises(ValueError, match="first target frame"):
        compute_prediction_residual(target + 1, prediction, valid_mask)


def test_top_k_is_exact_and_ties_prefer_lower_dimensions() -> None:
    values = np.zeros((3, 18), dtype=np.float32)
    values[1] = 1
    values[2, :4] = np.array([-3, 3, -2, 1], dtype=np.float32)
    residual = ResidualSequence(values, np.ones(3, dtype=np.bool_))

    selection = retain_top_k(residual, 2)

    assert selection.indices.tolist() == [[-1, -1], [0, 1], [0, 1]]
    assert selection.selection_mask.sum(axis=1).tolist() == [0, 2, 2]
    reconstructed = reconstruct_motion(np.zeros_like(values), selection)
    assert np.array_equal(reconstructed, selection.retained.values)


def test_top_k_can_rank_normalized_scores_but_retains_original_values() -> None:
    values = np.zeros((2, 18), dtype=np.float32)
    values[1, :2] = [10, 1]
    residual = ResidualSequence(values, np.ones(2, dtype=np.bool_))
    scores = normalize_residual(
        residual,
        np.array([10, 0.01, *([1] * 16)], dtype=np.float32),
    )

    selection = retain_top_k(residual, 1, scores=scores)

    assert selection.indices.tolist() == [[-1], [1]]
    assert selection.retained.values[1, 1] == 1
    assert selection.retained.values[1, 0] == 0


@pytest.mark.parametrize("k", [0, 18])
@pytest.mark.parametrize("method", ["top_k", "random_k"])
def test_selection_boundaries(k: int, method: str) -> None:
    residual = _residual_fixture()

    selection = (
        retain_top_k(residual, k) if method == "top_k" else retain_random_k(residual, k, seed=42)
    )

    assert selection.indices.shape == (4, k)
    assert selection.selection_mask.sum(axis=1).tolist() == [0, k, k, 0]
    if k == 0:
        assert np.count_nonzero(selection.retained.values) == 0
    else:
        assert np.array_equal(selection.retained.values, residual.values)


def test_random_k_uses_explicit_seed_and_exact_counts() -> None:
    residual = _residual_fixture()

    first = retain_random_k(residual, 5, seed=42)
    second = retain_random_k(residual, 5, seed=42)
    changed = retain_random_k(residual, 5, seed=43)

    assert np.array_equal(first.indices, second.indices)
    assert np.array_equal(first.selection_mask, second.selection_mask)
    assert not np.array_equal(first.indices, changed.indices)
    assert first.selection_mask.sum(axis=1).tolist() == [0, 5, 5, 0]
    with pytest.raises(TypeError, match="explicit integer"):
        retain_random_k(residual, 5, seed=np.int64(42))  # type: ignore[arg-type]


def test_original_and_normalized_metrics_are_per_dimension_and_masked() -> None:
    values = np.zeros((4, 18), dtype=np.float32)
    values[1] = 2
    values[2] = 4
    original = ResidualSequence(values, np.array([True, True, True, False]))
    normalized = normalize_residual(original, np.full(18, 2, dtype=np.float32))

    metrics = compute_per_dimension_metrics(original, normalized)

    assert metrics.original.l1.shape == (18,)
    assert np.allclose(metrics.original.l1, 2)
    assert np.allclose(metrics.original.rmse, np.sqrt(20 / 3))
    assert np.allclose(metrics.original.velocity_l1, 2)
    assert np.allclose(metrics.normalized.l1, 1)
    assert np.allclose(metrics.normalized.rmse, np.sqrt(5 / 3))
    assert np.allclose(metrics.normalized.velocity_l1, 1)


def test_energy_concentration_sorts_stably_and_reaches_one() -> None:
    values = np.zeros((4, 18), dtype=np.float32)
    values[1, 2] = 3
    values[2, 0] = 2
    values[2, 1] = 2
    residual = ResidualSequence(values, np.array([True, True, True, False]))

    concentration = compute_energy_concentration(residual)

    assert concentration.dimensions.sorted_indices[:3].tolist() == [2, 0, 1]
    assert concentration.dimensions.sorted_energy[:3].tolist() == [9, 4, 4]
    assert concentration.dimensions.cumulative_fraction[-1] == pytest.approx(1)
    assert concentration.frames.sorted_indices.tolist() == [1, 2, 0]
    assert concentration.frames.sorted_energy.tolist() == [9, 8, 0]
    assert concentration.frames.cumulative_fraction[-1] == pytest.approx(1)


def test_zero_energy_concentration_is_well_defined() -> None:
    residual = ResidualSequence(
        np.zeros((2, 18), dtype=np.float32),
        np.ones(2, dtype=np.bool_),
    )

    concentration = compute_energy_concentration(residual)

    assert np.all(concentration.dimensions.cumulative_fraction == 0)
    assert np.all(concentration.frames.cumulative_fraction == 0)


def test_selection_accounting_is_counts_not_a_bitrate_claim() -> None:
    selection = retain_top_k(_residual_fixture(), 3)

    accounting = selection_accounting(selection)

    assert DIMENSION_INDEX_BITS == 5
    assert accounting.eligible_frame_count == 2
    assert accounting.dense_value_count == 36
    assert accounting.retained_value_count == 6
    assert accounting.dimension_index_count == 6
    assert accounting.dimension_index_bits_per_value == 5
    assert accounting.dimension_index_bit_count == 30
    assert accounting.accounting_scope == "scalar_and_fixed_width_dimension_index_counts_only"


@pytest.mark.parametrize("k", [0, 18])
def test_empty_and_dense_accounting_need_no_adaptive_dimension_indices(k: int) -> None:
    accounting = selection_accounting(retain_top_k(_residual_fixture(), k))

    assert accounting.retained_value_count == accounting.eligible_frame_count * k
    assert accounting.dimension_index_count == 0
    assert accounting.dimension_index_bit_count == 0


def test_invalid_shapes_scales_and_k_are_rejected() -> None:
    residual = _residual_fixture()

    with pytest.raises(ValueError, match=r"shape \[T, 18\]"):
        compute_prediction_residual(
            np.zeros((2, 17), dtype=np.float32),
            np.zeros((2, 17), dtype=np.float32),
            np.ones(2, dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="boolean"):
        compute_prediction_residual(
            np.zeros((2, 18), dtype=np.float32),
            np.zeros((2, 18), dtype=np.float32),
            np.ones(2, dtype=np.int64),
        )
    with pytest.raises(ValueError, match="positive"):
        normalize_residual(residual, np.zeros(18, dtype=np.float32))
    with pytest.raises(ValueError, match=r"\[0, 18\]"):
        retain_top_k(residual, 19)
