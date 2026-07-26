"""Normalization and deterministic mouth-motion perturbations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from av_semcom.data.preprocessing import atomic_write_json
from av_semcom.models.motion.sequence import MotionSequence


@dataclass(frozen=True)
class MotionNormalizer:
    """Per-coordinate pilot normalization statistics for 18-D mouth motion."""

    mean: np.ndarray
    std: np.ndarray
    scope: str = "pilot_stats"

    def __post_init__(self) -> None:
        if self.mean.shape != (18,) or self.std.shape != (18,):
            raise ValueError("motion normalizer mean and std must have shape (18,)")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise ValueError("motion normalizer contains non-finite values")
        if np.any(self.std <= 0):
            raise ValueError("motion normalizer std must be positive")

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Normalize a ``[..., 18]`` motion tensor."""

        if values.shape[-1] != 18:
            raise ValueError("motion values must end with 18 coordinates")
        return ((values - self.mean) / self.std).astype(np.float32)

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        """Undo normalization."""

        if values.shape[-1] != 18:
            raise ValueError("motion values must end with 18 coordinates")
        return (values * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible statistics."""

        return {
            "scope": self.scope,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "dimension": 18,
        }


def fit_motion_normalizer(
    sequences: list[MotionSequence],
    *,
    minimum_std: float = 1e-6,
    scope: str = "pilot_stats",
) -> MotionNormalizer:
    """Fit per-coordinate normalization statistics for an explicit data scope."""

    if not sequences:
        raise ValueError("at least one motion sequence is required")
    if minimum_std <= 0:
        raise ValueError("minimum_std must be positive")
    if not scope:
        raise ValueError("scope must be non-empty")
    values = np.concatenate([sequence.lip_vector for sequence in sequences], axis=0)
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(minimum_std))
    return MotionNormalizer(mean=mean, std=std, scope=scope)


def save_motion_normalizer(path: Path, normalizer: MotionNormalizer) -> None:
    """Atomically save normalization statistics."""

    atomic_write_json(path, normalizer.to_dict())


def load_motion_normalizer(path: Path) -> MotionNormalizer:
    """Load normalization statistics from JSON."""

    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    return MotionNormalizer(
        mean=np.asarray(payload["mean"], dtype=np.float32),
        std=np.asarray(payload["std"], dtype=np.float32),
        scope=str(payload["scope"]),
    )


@dataclass(frozen=True)
class PerturbationCondition:
    """One reproducible motion sensitivity condition."""

    name: str
    family: str
    value: float | int | None = None
    seed: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible condition metadata."""

        return asdict(self)


def perturbation_parameter_name(family: str) -> str | None:
    """Return the physical meaning of ``value`` for one condition family."""

    return {
        "gaussian": "noise_standard_deviation",
        "quantization": "quantization_bits",
        "random_dropout": "keep_ratio",
        "magnitude_sparsity": "keep_ratio",
    }.get(family)


def default_perturbation_conditions() -> tuple[PerturbationCondition, ...]:
    """Return the fixed E2 sensitivity grid."""

    conditions: list[PerturbationCondition] = [
        PerturbationCondition("lip_only", "identity"),
        PerturbationCondition("frozen", "frozen"),
    ]
    conditions.extend(
        PerturbationCondition(f"gaussian_sigma_{sigma:g}", "gaussian", sigma, 42)
        for sigma in (0.05, 0.1, 0.2, 0.5)
    )
    conditions.extend(
        PerturbationCondition(f"quantization_{bits}bit", "quantization", bits)
        for bits in (8, 6, 4, 3, 2)
    )
    for keep_ratio in (0.75, 0.5, 0.25, 0.1):
        for seed in (42, 43, 44):
            conditions.append(
                PerturbationCondition(
                    f"random_keep_{keep_ratio:g}_seed_{seed}",
                    "random_dropout",
                    keep_ratio,
                    seed,
                )
            )
    conditions.extend(
        PerturbationCondition(
            f"magnitude_keep_{keep_ratio:g}",
            "magnitude_sparsity",
            keep_ratio,
        )
        for keep_ratio in (0.75, 0.5, 0.25, 0.1)
    )
    return tuple(conditions)


def _keep_count(keep_ratio: float) -> int:
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    return max(1, round(18 * keep_ratio))


def apply_perturbation(
    normalized_motion: np.ndarray,
    condition: PerturbationCondition,
    *,
    source_frame_index: int,
) -> np.ndarray:
    """Apply one perturbation in normalized coordinates.

    Random and magnitude masks retain exactly ``round(18 * keep_ratio)``
    coordinates per frame. The normalized value corresponding to raw zero is
    preserved at the source frame.
    """

    if normalized_motion.ndim != 2 or normalized_motion.shape[1] != 18:
        raise ValueError("normalized_motion must have shape [T, 18]")
    if not 0 <= source_frame_index < normalized_motion.shape[0]:
        raise ValueError("source_frame_index is outside normalized_motion")
    result = normalized_motion.astype(np.float32, copy=True)
    source_value = result[source_frame_index].copy()

    if condition.family == "identity":
        pass
    elif condition.family == "frozen":
        result[:] = source_value
    elif condition.family == "gaussian":
        if condition.value is None or condition.seed is None:
            raise ValueError("gaussian condition requires value and seed")
        rng = np.random.default_rng(condition.seed)
        result += rng.normal(0.0, float(condition.value), result.shape).astype(np.float32)
    elif condition.family == "quantization":
        if condition.value is None:
            raise ValueError("quantization condition requires a bit count")
        bits = int(condition.value)
        if bits < 1:
            raise ValueError("quantization bits must be positive")
        clipped = np.clip(result, -3.0, 3.0)
        levels = 2**bits
        step = 6.0 / (levels - 1)
        result = (np.round((clipped + 3.0) / step) * step - 3.0).astype(np.float32)
    elif condition.family == "random_dropout":
        if condition.value is None or condition.seed is None:
            raise ValueError("random_dropout condition requires value and seed")
        keep = _keep_count(float(condition.value))
        rng = np.random.default_rng(condition.seed)
        mask = np.zeros_like(result, dtype=np.bool_)
        for frame_index in range(result.shape[0]):
            indices = rng.choice(18, size=keep, replace=False)
            mask[frame_index, indices] = True
        result = np.where(mask, result, 0).astype(np.float32)
    elif condition.family == "magnitude_sparsity":
        if condition.value is None:
            raise ValueError("magnitude_sparsity condition requires a value")
        keep = _keep_count(float(condition.value))
        indices = np.argpartition(np.abs(result), -keep, axis=1)[:, -keep:]
        mask = np.zeros_like(result, dtype=np.bool_)
        np.put_along_axis(mask, indices, True, axis=1)
        result = np.where(mask, result, 0).astype(np.float32)
    else:
        raise ValueError(f"unsupported perturbation family: {condition.family}")

    result[source_frame_index] = source_value
    return result
