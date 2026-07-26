"""Configuration for the causal audio-to-mouth-motion baseline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from av_semcom.data.grid import GridSettings
from av_semcom.utils.config import ConfigError


@dataclass(frozen=True)
class AudioMotionSettings:
    """Resolved model, training, evaluation, and artifact settings."""

    data_settings: GridSettings
    motion_stats_path: Path
    output_root: Path
    mel_bins: int
    mel_steps_per_frame: int
    audio_projection_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float
    output_dim: int
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
    mixed_precision: bool
    deterministic: bool
    evaluation_splits: tuple[str, ...]
    baselines: tuple[str, ...]
    config: Mapping[str, Any]

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        data_settings: GridSettings,
    ) -> AudioMotionSettings:
        """Validate and resolve an E3 configuration."""

        model = _mapping(config, "model")
        training = _mapping(config, "training")
        evaluation = _mapping(config, "evaluation")
        motion = _mapping(config, "motion")

        if bool(model.get("bidirectional", False)):
            raise ConfigError("model.bidirectional must remain false for the causal baseline")
        output_dim = _positive_int(model, "output_dim", 18)
        if output_dim != 18:
            raise ConfigError("model.output_dim must be 18")
        mel_bins = _positive_int(model, "mel_bins", 80)
        mel_steps = _positive_int(model, "mel_steps_per_frame", 4)
        dropout = float(model.get("dropout", 0.1))
        if not 0 <= dropout < 1:
            raise ConfigError("model.dropout must be in [0, 1)")

        motion_output = _relative_data_path(
            motion.get("output_dir", "grid/processed/motion/liveportrait"),
            data_settings,
            "motion.output_dir",
        )
        stats_filename = motion.get("stats_filename", "train_stats.json")
        if not isinstance(stats_filename, str) or not stats_filename:
            raise ConfigError("motion.stats_filename must be a non-empty relative path")
        stats_path = (motion_output / stats_filename).resolve()
        if motion_output not in stats_path.parents:
            raise ConfigError("motion.stats_filename escapes motion.output_dir")

        output_raw = evaluation.get("output_dir", "outputs/audio_to_motion")
        if not isinstance(output_raw, str) or not output_raw:
            raise ConfigError("evaluation.output_dir must be a non-empty path")
        project_root = Path(__file__).resolve().parents[4]
        output_root = Path(output_raw)
        if not output_root.is_absolute():
            output_root = project_root / output_root

        seeds_raw = training.get("seeds", [42, 43, 44])
        if not isinstance(seeds_raw, list) or not seeds_raw:
            raise ConfigError("training.seeds must be a non-empty list")
        seeds = tuple(int(seed) for seed in seeds_raw)
        if any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ConfigError("training.seeds must be unique non-negative integers")

        splits_raw = evaluation.get("splits", ["validation", "test"])
        if not isinstance(splits_raw, list) or not splits_raw:
            raise ConfigError("evaluation.splits must be a non-empty list")
        splits = tuple(str(split) for split in splits_raw)
        if set(splits) != {"validation", "test"}:
            raise ConfigError("evaluation.splits must contain validation and test exactly")

        baselines_raw = evaluation.get(
            "baselines",
            ["zero_motion", "train_mean", "oracle_persistence"],
        )
        allowed_baselines = {"zero_motion", "train_mean", "oracle_persistence"}
        if not isinstance(baselines_raw, list) or set(baselines_raw) != allowed_baselines:
            raise ConfigError(
                "evaluation.baselines must contain zero_motion, train_mean, "
                "and oracle_persistence exactly"
            )

        return cls(
            data_settings=data_settings,
            motion_stats_path=stats_path,
            output_root=output_root.resolve(),
            mel_bins=mel_bins,
            mel_steps_per_frame=mel_steps,
            audio_projection_dim=_positive_int(model, "audio_projection_dim", 128),
            hidden_dim=_positive_int(model, "hidden_dim", 256),
            num_layers=_positive_int(model, "num_layers", 2),
            dropout=dropout,
            output_dim=output_dim,
            seeds=seeds,
            device=str(training.get("device", "cuda:0")),
            batch_size=_positive_int(training, "batch_size", 16),
            learning_rate=_positive_float(training, "learning_rate", 1e-3),
            weight_decay=_nonnegative_float(training, "weight_decay", 1e-4),
            max_epochs=_positive_int(training, "max_epochs", 100),
            early_stopping_patience=_positive_int(
                training,
                "early_stopping_patience",
                15,
            ),
            early_stopping_min_delta=_nonnegative_float(
                training,
                "early_stopping_min_delta",
                1e-4,
            ),
            gradient_clip_norm=_positive_float(training, "gradient_clip_norm", 1.0),
            num_workers=int(training.get("num_workers", 0)),
            mixed_precision=bool(training.get("mixed_precision", True)),
            deterministic=bool(training.get("deterministic", True)),
            evaluation_splits=splits,
            baselines=tuple(str(value) for value in baselines_raw),
            config=config,
        )


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} configuration must be a mapping")
    return value


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


def _relative_data_path(value: Any, settings: GridSettings, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{name} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute():
        raise ConfigError(f"{name} must be relative to DATA_ROOT")
    resolved = (settings.data_root / path).resolve()
    if settings.data_root not in resolved.parents:
        raise ConfigError(f"{name} escapes DATA_ROOT")
    return resolved
