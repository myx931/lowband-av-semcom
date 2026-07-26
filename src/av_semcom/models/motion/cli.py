"""Shared command-line parsing for E2 motion scripts."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from av_semcom.data.grid import GridSettings
from av_semcom.models.motion.config import MotionSettings
from av_semcom.utils.config import Config, ConfigError, load_yaml_config


def build_motion_parser(description: str) -> argparse.ArgumentParser:
    """Create a parser with safe data and resume overrides."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/motion/liveportrait.yaml"),
    )
    parser.add_argument("--speakers", nargs="+", help="Speaker IDs such as s1.")
    parser.add_argument("--max-samples", type=int, help="Maximum samples per speaker.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip matching artifacts; enabled by configuration by default.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace artifacts made with another configuration.",
    )
    return parser


def motion_settings_from_args(
    args: argparse.Namespace,
) -> tuple[Config, GridSettings, MotionSettings]:
    """Load one E2 configuration and apply command-line data overrides."""

    config: Config = deepcopy(load_yaml_config(args.config))
    data_raw = config.get("data")
    if not isinstance(data_raw, dict):
        raise ConfigError("data configuration must be a mapping")
    data: dict[str, Any] = data_raw
    if args.speakers is not None:
        data["speakers"] = args.speakers
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ConfigError("--max-samples must be positive")
        data["max_samples"] = args.max_samples
    if args.resume is not None:
        data["resume"] = args.resume
    data_settings = GridSettings.from_config(config)
    return config, data_settings, MotionSettings.from_config(config, data_settings)
