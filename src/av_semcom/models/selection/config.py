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
