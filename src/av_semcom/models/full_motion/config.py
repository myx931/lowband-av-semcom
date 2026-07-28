"""Configuration adapter for the matched full-motion JSCC control."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from av_semcom.models.jscc.config import JSCCSettings
from av_semcom.utils.config import ConfigError


def full_motion_jscc_settings(config: Mapping[str, Any]) -> JSCCSettings:
    """Reuse E5 settings while changing only representation identity and output."""

    raw = config.get("full_motion_jscc")
    if not isinstance(raw, Mapping):
        raise ConfigError("full_motion_jscc configuration must be a mapping")
    if raw.get("representation") != "train_standardized_full_18d_motion":
        raise ConfigError("full_motion_jscc representation must be full 18-D motion")
    output = raw.get("output_dir")
    if not isinstance(output, str) or not output:
        raise ConfigError("full_motion_jscc.output_dir must be a non-empty path")
    copied = deepcopy(dict(config))
    evaluation = copied.get("jscc_evaluation")
    model = copied.get("jscc_model")
    if not isinstance(evaluation, dict) or not isinstance(model, dict):
        raise ConfigError("full-motion control requires JSCC evaluation and model mappings")
    evaluation["output_dir"] = output
    model["name"] = "full_motion_mlp_jscc"
    settings = JSCCSettings.from_config(copied)
    expected = Path(output)
    if not expected.is_absolute():
        expected = Path(__file__).resolve().parents[4] / expected
    if settings.output_root != expected.resolve():
        raise ConfigError("full-motion output path resolution failed")
    return settings


def comparison_output_root(config: Mapping[str, Any]) -> Path:
    """Resolve the small matched-rate comparison output directory."""

    raw = config.get("full_motion_jscc")
    if not isinstance(raw, Mapping):
        raise ConfigError("full_motion_jscc configuration must be a mapping")
    output = raw.get("comparison_output_dir")
    if not isinstance(output, str) or not output:
        raise ConfigError("comparison_output_dir must be a non-empty path")
    path = Path(output)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[4] / path
    return path.resolve()
