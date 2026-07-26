"""Configuration resolution for motion extraction and sensitivity experiments."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from av_semcom.data.grid import GridSettings
from av_semcom.utils.config import ConfigError


@dataclass(frozen=True)
class MotionSettings:
    """Resolved E2 paths and frozen-backend settings."""

    data_settings: GridSettings
    output_root: Path
    stats_path: Path
    stats_split: str | None
    stats_scope: str
    backend: str
    backend_revision: str
    repository: Path
    model_root: Path | None
    device: str
    half_precision: bool
    stitching: bool
    reconstruction_batch_size: int
    experiment_root: Path
    save_sample_positions: tuple[int, ...]
    metric_workers: int
    config: Mapping[str, Any]

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        data_settings: GridSettings,
    ) -> MotionSettings:
        """Resolve repository, model, data, and experiment paths."""

        motion = config.get("motion")
        if not isinstance(motion, Mapping):
            raise ConfigError("motion configuration must be a mapping")

        output_relative = motion.get("output_dir", "grid/processed/motion/liveportrait")
        if not isinstance(output_relative, str) or not output_relative:
            raise ConfigError("motion.output_dir must be a non-empty relative path")
        output_path = Path(output_relative)
        if output_path.is_absolute():
            raise ConfigError("motion.output_dir must be relative to DATA_ROOT")
        output_root = (data_settings.data_root / output_path).resolve()
        if data_settings.data_root not in output_root.parents:
            raise ConfigError("motion.output_dir escapes DATA_ROOT")

        stats_filename = motion.get("stats_filename", "pilot_stats.json")
        if not isinstance(stats_filename, str) or not stats_filename:
            raise ConfigError("motion.stats_filename must be a non-empty relative path")
        stats_relative = Path(stats_filename)
        if stats_relative.is_absolute():
            raise ConfigError("motion.stats_filename must be relative to motion.output_dir")
        stats_path = (output_root / stats_relative).resolve()
        if stats_path != output_root and output_root not in stats_path.parents:
            raise ConfigError("motion.stats_filename escapes motion.output_dir")

        stats_split = motion.get("stats_split")
        if stats_split is not None and (not isinstance(stats_split, str) or not stats_split):
            raise ConfigError("motion.stats_split must be a non-empty string or null")
        stats_scope = motion.get("stats_scope", "pilot_stats")
        if not isinstance(stats_scope, str) or not stats_scope:
            raise ConfigError("motion.stats_scope must be a non-empty string")

        project_root = Path(__file__).resolve().parents[4]
        repository_raw = motion.get("repository", "third_party/LivePortrait")
        if not isinstance(repository_raw, str) or not repository_raw:
            raise ConfigError("motion.repository must be a non-empty path")
        repository = Path(repository_raw)
        if not repository.is_absolute():
            repository = project_root / repository

        model_root_env = motion.get("model_root_env", "MODEL_ROOT")
        if not isinstance(model_root_env, str) or not model_root_env:
            raise ConfigError("motion.model_root_env must name an environment variable")
        model_base = os.environ.get(model_root_env)
        model_root = (
            Path(model_base).expanduser().resolve() / "liveportrait" if model_base else None
        )

        experiment = config.get("experiment", {})
        if not isinstance(experiment, Mapping):
            raise ConfigError("experiment configuration must be a mapping")
        experiment_raw = experiment.get("output_dir", "outputs/motion_sensitivity")
        if not isinstance(experiment_raw, str) or not experiment_raw:
            raise ConfigError("experiment.output_dir must be a non-empty path")
        experiment_root = Path(experiment_raw)
        if not experiment_root.is_absolute():
            experiment_root = project_root / experiment_root

        positions_raw = experiment.get("save_sample_positions", [0, 9, 19])
        if not isinstance(positions_raw, list) or not all(
            isinstance(value, int) and value >= 0 for value in positions_raw
        ):
            raise ConfigError("experiment.save_sample_positions must be non-negative integers")
        reconstruction_batch_size = motion.get("reconstruction_batch_size", 16)
        if not isinstance(reconstruction_batch_size, int) or reconstruction_batch_size < 1:
            raise ConfigError("motion.reconstruction_batch_size must be a positive integer")
        metric_workers = experiment.get("metric_workers", 1)
        if not isinstance(metric_workers, int) or metric_workers < 1:
            raise ConfigError("experiment.metric_workers must be a positive integer")

        backend = motion.get("backend", "liveportrait")
        if backend not in {"liveportrait", "fake"}:
            raise ConfigError("motion.backend must be 'liveportrait' or 'fake'")
        revision = motion.get("backend_revision")
        if not isinstance(revision, str) or not revision:
            raise ConfigError("motion.backend_revision must be a non-empty string")

        return cls(
            data_settings=data_settings,
            output_root=output_root,
            stats_path=stats_path,
            stats_split=stats_split,
            stats_scope=stats_scope,
            backend=backend,
            backend_revision=revision,
            repository=repository.resolve(),
            model_root=model_root,
            device=str(motion.get("device", "cuda:0")),
            half_precision=bool(motion.get("half_precision", True)),
            stitching=bool(motion.get("stitching", True)),
            reconstruction_batch_size=reconstruction_batch_size,
            experiment_root=experiment_root.resolve(),
            save_sample_positions=tuple(positions_raw),
            metric_workers=metric_workers,
            config=config,
        )
