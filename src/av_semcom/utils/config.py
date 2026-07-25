"""YAML configuration loading with clear validation errors."""

from pathlib import Path
from typing import Any

import yaml

Config = dict[str, Any]


class ConfigError(ValueError):
    """Raised when a project configuration cannot be loaded safely."""


def load_yaml_config(path: str | Path) -> Config:
    """Load a YAML mapping from *path*.

    Args:
        path: YAML file to read.

    Returns:
        A mutable dictionary containing the parsed configuration.

    Raises:
        ConfigError: If the path is missing, unreadable, invalid YAML, or not a mapping.
    """

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")

    try:
        with config_path.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not load YAML configuration {config_path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ConfigError(f"Configuration root must be a YAML mapping: {config_path}")

    return loaded
