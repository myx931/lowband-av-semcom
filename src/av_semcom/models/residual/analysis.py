"""Pure NumPy analysis primitives for prediction residual experiments.

The accounting helpers in this module report scalar and index counts only.
They do not model a codec, entropy coding, packet headers, or bitrate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

MOTION_DIMENSIONS = 18
DIMENSION_INDEX_BITS = 5


FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]
MetricArray = NDArray[np.float64]


def _validate_motion(values: np.ndarray, *, name: str) -> FloatArray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != MOTION_DIMENSIONS:
        raise ValueError(f"{name} must have shape [T, {MOTION_DIMENSIONS}]")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one frame")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} must have a floating-point dtype")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.asarray(array, dtype=np.float32)


def _validate_valid_mask(valid_mask: np.ndarray, *, frame_count: int) -> BoolArray:
    mask = np.asarray(valid_mask)
    if mask.dtype != np.bool_ or mask.shape != (frame_count,):
        raise ValueError("valid_mask must be a boolean array with shape [T]")
    return np.asarray(mask, dtype=np.bool_)


@dataclass(frozen=True)
class ResidualSequence:
    """A source-relative ``[T, 18]`` prediction residual and its valid mask.

    Frame zero is the fixed reference-frame convention and therefore has a
    zero residual. Invalid frames are also represented by zero residuals so
    downstream selection cannot accidentally retain unusable values.
    """

    values: FloatArray
    valid_mask: BoolArray

    def __post_init__(self) -> None:
        values = _validate_motion(self.values, name="residual values")
        valid_mask = _validate_valid_mask(self.valid_mask, frame_count=values.shape[0])
        if not np.allclose(values[0], 0.0, rtol=0.0, atol=1e-7):
            raise ValueError("the first-frame residual must be zero")
        if np.any(values[~valid_mask] != 0):
            raise ValueError("invalid frames must have zero residual")
        object.__setattr__(self, "values", values.copy())
        object.__setattr__(self, "valid_mask", valid_mask.copy())

    @property
    def frame_count(self) -> int:
        """Return the number of frames."""

        return int(self.values.shape[0])

    @property
    def transmission_mask(self) -> BoolArray:
        """Return valid frames eligible for residual transmission.

        The reference frame is deliberately excluded because its residual is
        fixed to zero by convention.
        """

        mask = self.valid_mask.copy()
        mask[0] = False
        return mask


@dataclass(frozen=True)
class ResidualSelection:
    """A fixed-budget retained residual and its selected coordinates."""

    retained: ResidualSequence
    selection_mask: BoolArray
    indices: IntArray
    k: int
    method: str
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.k <= MOTION_DIMENSIONS:
            raise ValueError(f"k must be in [0, {MOTION_DIMENSIONS}]")
        expected_shape = (self.retained.frame_count, MOTION_DIMENSIONS)
        if self.selection_mask.dtype != np.bool_ or self.selection_mask.shape != expected_shape:
            raise ValueError(f"selection_mask must be boolean with shape {expected_shape}")
        if self.indices.dtype != np.int64 or self.indices.shape != (
            self.retained.frame_count,
            self.k,
        ):
            raise ValueError("indices must have shape [T, k] and dtype int64")
        transmission_mask = self.retained.transmission_mask
        selected_counts = self.selection_mask.sum(axis=1)
        if np.any(selected_counts[transmission_mask] != self.k):
            raise ValueError("each eligible frame must select exactly k dimensions")
        if np.any(selected_counts[~transmission_mask] != 0):
            raise ValueError("reference and invalid frames cannot select residual dimensions")
        if self.k and np.any(self.indices[transmission_mask] < 0):
            raise ValueError("eligible-frame indices must be non-negative")
        if np.any(self.indices[~transmission_mask] != -1):
            raise ValueError("ineligible-frame indices must use the -1 sentinel")
        if np.any(self.retained.values[~self.selection_mask] != 0):
            raise ValueError("unselected residual values must be zero")


@dataclass(frozen=True)
class PerDimensionMetrics:
    """Per-dimension residual magnitude and temporal-change statistics."""

    l1: MetricArray
    rmse: MetricArray
    velocity_l1: MetricArray

    def to_dict(self) -> dict[str, list[float]]:
        """Return JSON-compatible metric vectors."""

        return {
            "l1": self.l1.tolist(),
            "rmse": self.rmse.tolist(),
            "velocity_l1": self.velocity_l1.tolist(),
        }


@dataclass(frozen=True)
class ResidualMetrics:
    """Residual statistics in original and train-normalized coordinates."""

    original: PerDimensionMetrics
    normalized: PerDimensionMetrics

    def to_dict(self) -> dict[str, dict[str, list[float]]]:
        """Return JSON-compatible metrics."""

        return {
            "original": self.original.to_dict(),
            "normalized": self.normalized.to_dict(),
        }


@dataclass(frozen=True)
class ConcentrationCurve:
    """Descending energy order and its cumulative fraction."""

    sorted_indices: IntArray
    sorted_energy: MetricArray
    cumulative_fraction: MetricArray

    def to_dict(self) -> dict[str, list[int] | list[float]]:
        """Return a JSON-compatible concentration curve."""

        return {
            "sorted_indices": self.sorted_indices.tolist(),
            "sorted_energy": self.sorted_energy.tolist(),
            "cumulative_fraction": self.cumulative_fraction.tolist(),
        }


@dataclass(frozen=True)
class ResidualConcentration:
    """Energy concentration across dimensions and valid frames."""

    dimensions: ConcentrationCurve
    frames: ConcentrationCurve

    def to_dict(self) -> dict[str, dict[str, list[int] | list[float]]]:
        """Return JSON-compatible concentration curves."""

        return {
            "dimensions": self.dimensions.to_dict(),
            "frames": self.frames.to_dict(),
        }


@dataclass(frozen=True)
class SelectionAccounting:
    """Symbol counts for one retained-residual selection.

    ``dimension_index_bit_count`` is only the fixed-width coordinate-index
    overhead implied by 18 dimensions for sparse ``0 < k < 18`` selections.
    Dense ``k=18`` values have a fixed dimension order and need no adaptive
    indices; ``k=0`` has no values to index. This is not a codec or bitrate
    estimate.
    """

    frame_count: int
    eligible_frame_count: int
    dimensions_per_frame: int
    dense_value_count: int
    retained_value_count: int
    dimension_index_count: int
    dimension_index_bits_per_value: int
    dimension_index_bit_count: int
    accounting_scope: str = "scalar_and_fixed_width_dimension_index_counts_only"

    def to_dict(self) -> dict[str, int | str]:
        """Return JSON-compatible count metadata."""

        return asdict(self)


def compute_prediction_residual(
    target: np.ndarray,
    prediction: np.ndarray,
    valid_mask: np.ndarray,
) -> ResidualSequence:
    """Compute ``target - prediction`` while preserving frame/mask conventions."""

    target_values = _validate_motion(target, name="target")
    prediction_values = _validate_motion(prediction, name="prediction")
    if target_values.shape != prediction_values.shape:
        raise ValueError("target and prediction must share shape [T, 18]")
    mask = _validate_valid_mask(valid_mask, frame_count=target_values.shape[0])
    if not np.allclose(target_values[0], 0.0, rtol=0.0, atol=1e-7):
        raise ValueError("the first target frame must be zero")
    if not np.allclose(prediction_values[0], 0.0, rtol=0.0, atol=1e-7):
        raise ValueError("the first prediction frame must be zero")
    residual = (target_values - prediction_values).astype(np.float32)
    residual[~mask] = 0.0
    residual[0] = 0.0
    return ResidualSequence(values=residual, valid_mask=mask)


def normalize_residual(
    residual: ResidualSequence,
    motion_std: np.ndarray,
) -> ResidualSequence:
    """Normalize residual coordinates with frozen train-set motion scales.

    A motion mean is intentionally unnecessary: subtracting target and
    prediction cancels the shared normalization mean.
    """

    std = np.asarray(motion_std)
    if std.shape != (MOTION_DIMENSIONS,) or not np.issubdtype(std.dtype, np.floating):
        raise ValueError(f"motion_std must be floating point with shape ({MOTION_DIMENSIONS},)")
    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("motion_std must contain finite positive values")
    values = (residual.values / std.astype(np.float32)).astype(np.float32)
    return ResidualSequence(values=values, valid_mask=residual.valid_mask)


def reconstruct_motion(
    prediction: np.ndarray,
    retained_residual: ResidualSequence | ResidualSelection,
) -> FloatArray:
    """Reconstruct motion as prediction plus a full or retained residual."""

    prediction_values = _validate_motion(prediction, name="prediction")
    sequence = (
        retained_residual.retained
        if isinstance(retained_residual, ResidualSelection)
        else retained_residual
    )
    if prediction_values.shape != sequence.values.shape:
        raise ValueError("prediction and residual must share shape [T, 18]")
    if not np.allclose(prediction_values[0], 0.0, rtol=0.0, atol=1e-7):
        raise ValueError("the first prediction frame must be zero")
    return (prediction_values + sequence.values).astype(np.float32)


def retain_top_k(
    residual: ResidualSequence,
    k: int,
    *,
    scores: ResidualSequence | None = None,
) -> ResidualSelection:
    """Retain the per-frame largest scores with deterministic low-index ties.

    By default the residual magnitude is its own score. Supplying, for example,
    the corresponding train-normalized residual as ``scores`` changes only the
    ranking: returned values always remain in the original ``residual`` space.
    """

    _validate_k(k)
    ranking = residual if scores is None else scores
    if ranking.values.shape != residual.values.shape:
        raise ValueError("scores and residual must share shape [T, 18]")
    if not np.array_equal(ranking.valid_mask, residual.valid_mask):
        raise ValueError("scores and residual must share a valid mask")
    selection_mask = np.zeros_like(residual.values, dtype=np.bool_)
    indices = np.full((residual.frame_count, k), -1, dtype=np.int64)
    for frame_index in np.flatnonzero(residual.transmission_mask):
        # Stable sorting makes equal magnitudes prefer the lower dimension index.
        selected = np.argsort(-np.abs(ranking.values[frame_index]), kind="stable")[:k]
        indices[frame_index] = selected
        selection_mask[frame_index, selected] = True
    return _build_selection(
        residual,
        selection_mask,
        indices,
        k=k,
        method="magnitude_top_k",
    )


def retain_random_k(
    residual: ResidualSequence,
    k: int,
    *,
    seed: int,
) -> ResidualSelection:
    """Retain exactly ``k`` uniformly sampled dimensions per eligible frame."""

    _validate_k(k)
    if not isinstance(seed, int):
        raise TypeError("seed must be an explicit integer")
    rng = np.random.default_rng(seed)
    selection_mask = np.zeros_like(residual.values, dtype=np.bool_)
    indices = np.full((residual.frame_count, k), -1, dtype=np.int64)
    for frame_index in np.flatnonzero(residual.transmission_mask):
        selected = np.sort(rng.choice(MOTION_DIMENSIONS, size=k, replace=False)).astype(np.int64)
        indices[frame_index] = selected
        selection_mask[frame_index, selected] = True
    return _build_selection(
        residual,
        selection_mask,
        indices,
        k=k,
        method="random_k",
        seed=seed,
    )


def compute_per_dimension_metrics(
    original_residual: ResidualSequence,
    normalized_residual: ResidualSequence,
) -> ResidualMetrics:
    """Compute per-dimension L1, RMSE, and velocity L1 in both spaces."""

    if original_residual.values.shape != normalized_residual.values.shape:
        raise ValueError("original and normalized residuals must share shape [T, 18]")
    if not np.array_equal(original_residual.valid_mask, normalized_residual.valid_mask):
        raise ValueError("original and normalized residuals must share a valid mask")
    return ResidualMetrics(
        original=_metrics_for_values(original_residual),
        normalized=_metrics_for_values(normalized_residual),
    )


def compute_energy_concentration(residual: ResidualSequence) -> ResidualConcentration:
    """Compute descending squared-energy concentration over dimensions and frames."""

    valid_values = residual.values[residual.valid_mask].astype(np.float64)
    dimension_energy = np.square(valid_values).sum(axis=0)
    valid_frame_indices = np.flatnonzero(residual.valid_mask).astype(np.int64)
    frame_energy = np.square(valid_values).sum(axis=1)
    return ResidualConcentration(
        dimensions=_concentration_curve(
            np.arange(MOTION_DIMENSIONS, dtype=np.int64),
            dimension_energy,
        ),
        frames=_concentration_curve(valid_frame_indices, frame_energy),
    )


def selection_accounting(selection: ResidualSelection) -> SelectionAccounting:
    """Count retained scalar values and fixed-width dimension-index overhead."""

    eligible_frame_count = int(selection.retained.transmission_mask.sum())
    retained_value_count = int(selection.selection_mask.sum())
    dense_value_count = eligible_frame_count * MOTION_DIMENSIONS
    dimension_index_count = retained_value_count if 0 < selection.k < MOTION_DIMENSIONS else 0
    return SelectionAccounting(
        frame_count=selection.retained.frame_count,
        eligible_frame_count=eligible_frame_count,
        dimensions_per_frame=MOTION_DIMENSIONS,
        dense_value_count=dense_value_count,
        retained_value_count=retained_value_count,
        dimension_index_count=dimension_index_count,
        dimension_index_bits_per_value=DIMENSION_INDEX_BITS,
        dimension_index_bit_count=dimension_index_count * DIMENSION_INDEX_BITS,
    )


def _validate_k(k: int) -> None:
    if not isinstance(k, int):
        raise TypeError("k must be an integer")
    if not 0 <= k <= MOTION_DIMENSIONS:
        raise ValueError(f"k must be in [0, {MOTION_DIMENSIONS}]")


def _build_selection(
    residual: ResidualSequence,
    selection_mask: BoolArray,
    indices: IntArray,
    *,
    k: int,
    method: str,
    seed: int | None = None,
) -> ResidualSelection:
    retained_values = np.where(selection_mask, residual.values, 0.0).astype(np.float32)
    retained = ResidualSequence(values=retained_values, valid_mask=residual.valid_mask)
    return ResidualSelection(
        retained=retained,
        selection_mask=selection_mask,
        indices=indices,
        k=k,
        method=method,
        seed=seed,
    )


def _metrics_for_values(residual: ResidualSequence) -> PerDimensionMetrics:
    values = residual.values.astype(np.float64)
    valid_values = values[residual.valid_mask]
    if valid_values.shape[0] == 0:
        raise ValueError("at least one valid residual frame is required")
    valid_velocity_pairs = residual.valid_mask[1:] & residual.valid_mask[:-1]
    velocity = np.diff(values, axis=0)[valid_velocity_pairs]
    velocity_l1 = (
        np.abs(velocity).mean(axis=0)
        if velocity.shape[0]
        else np.zeros(MOTION_DIMENSIONS, dtype=np.float64)
    )
    return PerDimensionMetrics(
        l1=np.abs(valid_values).mean(axis=0),
        rmse=np.sqrt(np.square(valid_values).mean(axis=0)),
        velocity_l1=velocity_l1,
    )


def _concentration_curve(indices: IntArray, energy: MetricArray) -> ConcentrationCurve:
    order = np.argsort(-energy, kind="stable")
    sorted_indices = indices[order]
    sorted_energy = energy[order].astype(np.float64)
    total = float(sorted_energy.sum())
    cumulative = (
        np.cumsum(sorted_energy, dtype=np.float64) / total
        if total > 0
        else np.zeros_like(sorted_energy, dtype=np.float64)
    )
    return ConcentrationCurve(
        sorted_indices=sorted_indices,
        sorted_energy=sorted_energy,
        cumulative_fraction=cumulative,
    )
