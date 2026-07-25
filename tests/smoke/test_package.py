"""Fast smoke tests for the package and baseline configuration."""

from pathlib import Path

import pytest

import av_semcom
from av_semcom.utils import load_yaml_config


@pytest.mark.smoke
def test_package_import_and_baseline_config() -> None:
    project_root = Path(__file__).parents[2]
    config = load_yaml_config(project_root / "configs/experiment/baseline.yaml")

    assert av_semcom.__version__ == "0.1.0"
    assert config["experiment"]["seed"] == 42
    assert config["data"]["dataset"] == "grid"
