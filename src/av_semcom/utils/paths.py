"""Resolve configurable project paths without machine-specific constants."""

import os
from pathlib import Path

from av_semcom.utils.config import ConfigError


def resolve_data_root(configured_root: str | Path | None = None) -> Path:
    """Resolve the dataset root from config, falling back to ``DATA_ROOT``.

    The function does not create the directory because dataset acquisition is a
    deliberate, user-controlled action.
    """

    raw_root = configured_root or os.environ.get("DATA_ROOT")
    if raw_root is None or not str(raw_root).strip():
        raise ConfigError("Data root is not configured. Set data.root in YAML or export DATA_ROOT.")
    return Path(raw_root).expanduser().resolve()


def resolve_output_root(
    configured_root: str | Path = "outputs",
    *,
    project_root: str | Path | None = None,
) -> Path:
    """Resolve an output root relative to the project when it is not absolute."""

    output_root = Path(configured_root).expanduser()
    if output_root.is_absolute():
        return output_root.resolve()

    base = Path(project_root).expanduser() if project_root is not None else Path.cwd()
    return (base / output_root).resolve()
