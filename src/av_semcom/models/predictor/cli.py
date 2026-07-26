"""Shared command-line parsing for E3 predictor experiments."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from av_semcom.data.grid import GridSettings
from av_semcom.models.motion.config import MotionSettings
from av_semcom.models.predictor.config import AudioMotionSettings
from av_semcom.utils.config import Config, ConfigError, load_yaml_config


def build_predictor_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/audio_to_motion_gru.yaml"),
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a matching run directory and skip completed seeds.",
    )
    return parser


def predictor_settings_from_args(
    args: argparse.Namespace,
) -> tuple[Config, GridSettings, AudioMotionSettings, MotionSettings]:
    config: Config = deepcopy(load_yaml_config(args.config))
    data_raw = config.get("data")
    if not isinstance(data_raw, dict):
        raise ConfigError("data configuration must be a mapping")
    data_settings = GridSettings.from_config(config)
    predictor = AudioMotionSettings.from_config(config, data_settings)
    motion = MotionSettings.from_config(config, data_settings)
    return config, data_settings, predictor, motion
