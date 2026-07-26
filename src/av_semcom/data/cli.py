"""Shared command-line parsing for GRID data scripts."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from av_semcom.data.grid import GridSettings
from av_semcom.utils.config import Config, ConfigError, load_yaml_config


def build_data_parser(description: str) -> argparse.ArgumentParser:
    """Create a parser with the common GRID subset options."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=Path("configs/data/grid.yaml"))
    parser.add_argument("--speakers", nargs="+", help="Speaker IDs such as s1 s2 s3.")
    parser.add_argument("--max-samples", type=int, help="Maximum paired samples per speaker.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip matching artifacts; enabled by configuration by default.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace artifacts even when their configuration fingerprint differs.",
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> GridSettings:
    """Load configuration and apply safe command-line overrides."""

    loaded = load_yaml_config(args.config)
    config: Config = deepcopy(loaded)
    data: dict[str, Any]
    if "data" in config:
        raw_data = config["data"]
        if not isinstance(raw_data, dict):
            raise ConfigError("data configuration must be a mapping")
        data = raw_data
    else:
        data = config
    if args.speakers is not None:
        data["speakers"] = args.speakers
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ConfigError("--max-samples must be positive")
        data["max_samples"] = args.max_samples
    if args.resume is not None:
        data["resume"] = args.resume
    return GridSettings.from_config(config)
