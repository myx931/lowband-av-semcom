"""Tests for read-only reconstruction progress inspection."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from av_semcom.utils import run_progress


def test_inspect_reconstruction_progress_and_eta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reconstruction = tmp_path / "reconstruction"
    samples = reconstruction / "samples"
    samples.mkdir(parents=True)
    (reconstruction / "runtime.json").write_text(
        json.dumps({"evaluation_sample_count": 4}), encoding="utf-8"
    )
    first = samples / "first.json"
    second = samples / "second.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    os.utime(first, (100.0, 100.0))
    os.utime(second, (160.0, 160.0))
    monkeypatch.setattr(run_progress, "_reconstruction_process_running", lambda _: True)
    monkeypatch.setattr(run_progress, "_gpu_status", lambda: "fake gpu")

    progress = run_progress.inspect_reconstruction_progress(tmp_path)

    assert progress.status == "running"
    assert progress.completed_samples == 2
    assert progress.total_samples == 4
    assert progress.percent_complete == 50.0
    assert progress.samples_per_minute == 1.0
    assert progress.eta_seconds == 120.0
    assert progress.gpu_status == "fake gpu"


def test_complete_progress_has_zero_eta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reconstruction = tmp_path / "reconstruction"
    (reconstruction / "samples").mkdir(parents=True)
    (reconstruction / "runtime.json").write_text(
        json.dumps({"evaluation_sample_count": 0}), encoding="utf-8"
    )
    (reconstruction / "complete.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(run_progress, "_gpu_status", lambda: None)

    progress = run_progress.inspect_reconstruction_progress(tmp_path)

    assert progress.status == "complete"
    assert progress.eta_seconds == 0.0
    assert run_progress.format_duration(progress.eta_seconds) == "00:00:00"
