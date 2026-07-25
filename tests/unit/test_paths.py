"""Tests for configuration-driven path resolution."""

from pathlib import Path

import pytest

from av_semcom.utils.config import ConfigError
from av_semcom.utils.paths import resolve_data_root, resolve_output_root


def test_data_root_uses_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    assert resolve_data_root() == tmp_path.resolve()


def test_configured_data_root_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configured = tmp_path / "configured"
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "environment"))

    assert resolve_data_root(configured) == configured.resolve()


def test_missing_data_root_has_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATA_ROOT", raising=False)

    with pytest.raises(ConfigError, match="DATA_ROOT"):
        resolve_data_root()


def test_relative_output_root_is_project_relative(tmp_path: Path) -> None:
    assert resolve_output_root("outputs", project_root=tmp_path) == (tmp_path / "outputs").resolve()
