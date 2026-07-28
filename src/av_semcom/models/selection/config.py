"""Configuration for the validation-only E6 channel gate baseline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from av_semcom.models.jscc.config import JSCCSettings
from av_semcom.utils.config import ConfigError


@dataclass(frozen=True)
class ChannelGateSettings:
    """Resolved settings for a global hard SNR gate per channel budget."""

    output_root: Path
    validation_snr_db: tuple[float, ...]
    noise_seeds: tuple[int, ...]
    primary_metric: str
    minimum_relative_improvement: float
    config: Mapping[str, Any]

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        jscc: JSCCSettings,
    ) -> ChannelGateSettings:
        """Validate a gate protocol that cannot tune on the E5 test grid."""

        raw = config.get("channel_gate")
        if not isinstance(raw, Mapping):
            raise ConfigError("channel_gate configuration must be a mapping")
        snr_raw = raw.get("validation_snr_db")
        if not isinstance(snr_raw, list) or not snr_raw:
            raise ConfigError("channel_gate.validation_snr_db must be a non-empty list")
        validation_snr = tuple(float(value) for value in snr_raw)
        if validation_snr != tuple(sorted(set(validation_snr))):
            raise ConfigError("channel_gate.validation_snr_db must be sorted and unique")
        if set(validation_snr) & set(jscc.test_snr_db):
            raise ConfigError(
                "channel_gate validation SNR grid must be disjoint from the E5 test grid"
            )

        noise_raw = raw.get("noise_seeds")
        if not isinstance(noise_raw, list) or not noise_raw:
            raise ConfigError("channel_gate.noise_seeds must be a non-empty list")
        noise_seeds = tuple(int(value) for value in noise_raw)
        if noise_seeds != jscc.noise_seeds:
            raise ConfigError("channel_gate.noise_seeds must equal jscc_evaluation.noise_seeds")

        primary_metric = str(raw.get("primary_metric", "l1"))
        if primary_metric != "l1":
            raise ConfigError("channel_gate.primary_metric currently supports only l1")
        minimum_improvement = float(raw.get("minimum_relative_improvement", 0.0))
        if not 0.0 <= minimum_improvement < 1.0:
            raise ConfigError("channel_gate.minimum_relative_improvement must be in [0,1)")

        output_raw = raw.get("output_dir", "outputs/channel_gate")
        if not isinstance(output_raw, str) or not output_raw:
            raise ConfigError("channel_gate.output_dir must be a non-empty path")
        output_root = Path(output_raw)
        if not output_root.is_absolute():
            output_root = Path(__file__).resolve().parents[4] / output_root
        return cls(
            output_root=output_root.resolve(),
            validation_snr_db=validation_snr,
            noise_seeds=noise_seeds,
            primary_metric=primary_metric,
            minimum_relative_improvement=minimum_improvement,
            config=dict(raw),
        )


@dataclass(frozen=True)
class ResidualScorerSettings:
    """Resolved E6 hard Top-K scorer training and evaluation settings."""

    output_root: Path
    budgets: Mapping[int, int]
    hidden_dim: int
    temperature: float
    velocity_weight: float
    seeds: tuple[int, ...]
    device: str
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    gradient_clip_norm: float
    num_workers: int
    deterministic: bool
    train_snr_min_db: float
    train_snr_max_db: float
    validation_snr_db: tuple[float, ...]
    noise_seeds: tuple[int, ...]
    random_seeds: tuple[int, ...]
    config: Mapping[str, Any]

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        jscc: JSCCSettings,
    ) -> ResidualScorerSettings:
        """Validate the deliberately small scorer experiment surface."""

        raw = config.get("residual_scorer")
        if not isinstance(raw, Mapping):
            raise ConfigError("residual_scorer configuration must be a mapping")
        budget_raw = raw.get("budgets_by_channel_use")
        if not isinstance(budget_raw, Mapping):
            raise ConfigError("residual_scorer.budgets_by_channel_use must be a mapping")
        budgets = {int(key): int(value) for key, value in budget_raw.items()}
        if set(budgets) != set(jscc.channel_uses):
            raise ConfigError("residual scorer budgets must cover every configured C")
        if any(budgets[channel_uses] != 2 * channel_uses for channel_uses in budgets):
            raise ConfigError("residual scorer protocol requires K=2C")
        if any(not 0 < value < 18 for value in budgets.values()):
            raise ConfigError("residual scorer budgets must be in [1,17]")

        validation_raw = raw.get("validation_snr_db")
        if not isinstance(validation_raw, list) or not validation_raw:
            raise ConfigError("residual_scorer.validation_snr_db must be a non-empty list")
        validation_snr = tuple(float(value) for value in validation_raw)
        if validation_snr != tuple(sorted(set(validation_snr))):
            raise ConfigError("residual_scorer.validation_snr_db must be sorted and unique")
        if set(validation_snr) & set(jscc.test_snr_db):
            raise ConfigError("residual scorer validation and test SNR grids must be disjoint")

        noise_seeds = _positive_seed_tuple(raw, "noise_seeds")
        if noise_seeds != jscc.noise_seeds:
            raise ConfigError("residual scorer noise seeds must equal E5 noise seeds")
        random_seeds = _positive_seed_tuple(raw, "random_seeds")
        seeds = _positive_seed_tuple(raw, "seeds")
        train_min = float(raw.get("snr_min_db", jscc.train_snr_min_db))
        train_max = float(raw.get("snr_max_db", jscc.train_snr_max_db))
        if not train_min < train_max:
            raise ConfigError("residual scorer train SNR range must have min < max")
        if any(not train_min <= value <= train_max for value in validation_snr):
            raise ConfigError("residual scorer validation SNR must be in its train range")

        output_raw = raw.get("output_dir", "outputs/residual_scorer")
        if not isinstance(output_raw, str) or not output_raw:
            raise ConfigError("residual_scorer.output_dir must be a non-empty path")
        output_root = Path(output_raw)
        if not output_root.is_absolute():
            output_root = Path(__file__).resolve().parents[4] / output_root
        num_workers = int(raw.get("num_workers", 0))
        if num_workers < 0:
            raise ConfigError("residual_scorer.num_workers must be non-negative")
        return cls(
            output_root=output_root.resolve(),
            budgets=budgets,
            hidden_dim=_positive_int(raw, "hidden_dim", 64),
            temperature=_positive_float(raw, "temperature", 1.0),
            velocity_weight=_nonnegative_float(raw, "velocity_weight", 0.5),
            seeds=seeds,
            device=str(raw.get("device", jscc.device)),
            batch_size=_positive_int(raw, "batch_size", 32),
            learning_rate=_positive_float(raw, "learning_rate", 1e-3),
            weight_decay=_nonnegative_float(raw, "weight_decay", 1e-4),
            max_epochs=_positive_int(raw, "max_epochs", 50),
            early_stopping_patience=_positive_int(
                raw,
                "early_stopping_patience",
                10,
            ),
            early_stopping_min_delta=_nonnegative_float(
                raw,
                "early_stopping_min_delta",
                1e-6,
            ),
            gradient_clip_norm=_positive_float(raw, "gradient_clip_norm", 1.0),
            num_workers=num_workers,
            deterministic=bool(raw.get("deterministic", True)),
            train_snr_min_db=train_min,
            train_snr_max_db=train_max,
            validation_snr_db=validation_snr,
            noise_seeds=noise_seeds,
            random_seeds=random_seeds,
            config=dict(raw),
        )


def _positive_int(config: Mapping[str, Any], key: str, default: int) -> int:
    value = int(config.get(key, default))
    if value <= 0:
        raise ConfigError(f"{key} must be positive")
    return value


def _positive_float(config: Mapping[str, Any], key: str, default: float) -> float:
    value = float(config.get(key, default))
    if value <= 0:
        raise ConfigError(f"{key} must be positive")
    return value


def _nonnegative_float(config: Mapping[str, Any], key: str, default: float) -> float:
    value = float(config.get(key, default))
    if value < 0:
        raise ConfigError(f"{key} must be non-negative")
    return value


def _positive_seed_tuple(config: Mapping[str, Any], key: str) -> tuple[int, ...]:
    raw = config.get(key)
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"residual_scorer.{key} must be a non-empty list")
    values = tuple(int(value) for value in raw)
    if values != tuple(sorted(set(values))) or any(value < 0 for value in values):
        raise ConfigError(f"residual_scorer.{key} must be sorted, unique, and non-negative")
    return values
