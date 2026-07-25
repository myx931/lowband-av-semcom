"""Tests for YAML configuration loading."""

from pathlib import Path

import pytest

from av_semcom.utils.config import ConfigError, load_yaml_config


def test_load_yaml_config_returns_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("experiment:\n  seed: 42\n", encoding="utf-8")

    config = load_yaml_config(config_path)

    assert config["experiment"]["seed"] == 42


def test_load_yaml_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        load_yaml_config(tmp_path / "missing.yaml")


def test_load_yaml_config_rejects_non_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "list.yaml"
    config_path.write_text("- invalid\n- root\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="must be a YAML mapping"):
        load_yaml_config(config_path)
