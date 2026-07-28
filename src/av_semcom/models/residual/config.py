"""Configuration for the prediction-residual baseline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from av_semcom.utils.config import ConfigError


@dataclass(frozen=True)
class ResidualSettings:
    """Resolved fixed-budget E4 experiment settings."""

    output_root: Path
    budgets: tuple[int, ...]
    random_seeds: tuple[int, ...]
    reconstruction_budgets: tuple[int, ...]
    selection_spaces: tuple[str, ...]
    value_storage_bits: int
    dimension_index_bits: int
    metric_workers: int
    representative_positions: tuple[str, ...]
    config: Mapping[str, Any]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ResidualSettings:
        """Validate residual budgets and accounting conventions."""

        residual = _mapping(config, "residual")
        experiment = _mapping(config, "experiment")
        spaces_raw = residual.get("selection_spaces", ["raw", "normalized"])
        if not isinstance(spaces_raw, list):
            raise ConfigError("residual.selection_spaces must be a list")
        selection_spaces = tuple(str(value) for value in spaces_raw)
        if selection_spaces != ("raw", "normalized"):
            raise ConfigError("residual.selection_spaces must contain raw then normalized")
        budgets = _unique_ints(residual, "budgets")
        if budgets[0] != 0 or budgets[-1] != 18 or any(not 0 <= k <= 18 for k in budgets):
            raise ConfigError("residual.budgets must be sorted, unique, and span 0 through 18")
        reconstruction_budgets = _unique_ints(residual, "reconstruction_budgets")
        if not set(reconstruction_budgets).issubset(budgets):
            raise ConfigError("reconstruction_budgets must be a subset of budgets")
        random_seeds = _unique_ints(residual, "random_seeds")
        if any(seed < 0 for seed in random_seeds):
            raise ConfigError("random seeds must be non-negative")
        value_storage_bits = int(residual.get("value_storage_bits", 32))
        dimension_index_bits = int(residual.get("dimension_index_bits", 5))
        if value_storage_bits <= 0:
            raise ConfigError("residual.value_storage_bits must be positive")
        if dimension_index_bits != 5:
            raise ConfigError("18 dimensions require exactly 5 index bits")
        output_raw = experiment.get("output_dir", "outputs/residual_baseline")
        if not isinstance(output_raw, str) or not output_raw:
            raise ConfigError("experiment.output_dir must be a non-empty path")
        output_root = Path(output_raw)
        if not output_root.is_absolute():
            output_root = Path(__file__).resolve().parents[4] / output_root
        metric_workers = int(experiment.get("metric_workers", 1))
        if metric_workers < 1:
            raise ConfigError("experiment.metric_workers must be positive")
        positions_raw = residual.get(
            "representative_positions",
            ["first", "middle", "last"],
        )
        if positions_raw != ["first", "middle", "last"]:
            raise ConfigError("residual.representative_positions must contain first, middle, last")
        return cls(
            output_root=output_root.resolve(),
            budgets=budgets,
            random_seeds=random_seeds,
            reconstruction_budgets=reconstruction_budgets,
            selection_spaces=selection_spaces,
            value_storage_bits=value_storage_bits,
            dimension_index_bits=dimension_index_bits,
            metric_workers=metric_workers,
            representative_positions=tuple(str(value) for value in positions_raw),
            config=config,
        )


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} configuration must be a mapping")
    return value


def _unique_ints(config: Mapping[str, Any], key: str) -> tuple[int, ...]:
    raw = config.get(key)
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{key} must be a non-empty list")
    values = tuple(int(value) for value in raw)
    if values != tuple(sorted(set(values))):
        raise ConfigError(f"{key} must be sorted and unique")
    return values
