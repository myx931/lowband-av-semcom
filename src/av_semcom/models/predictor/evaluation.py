"""Independent validation of E3 checkpoints, predictions, and metric rows."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from av_semcom.data.grid import GridSettings, read_manifest
from av_semcom.data.preprocessing import atomic_write_json
from av_semcom.models.predictor.artifacts import load_prediction
from av_semcom.models.predictor.config import AudioMotionSettings
from av_semcom.models.predictor.data import select_predictor_samples


def validate_audio_motion_run(
    run_dir: Path,
    settings: AudioMotionSettings,
    data_settings: GridSettings,
) -> dict[str, Any]:
    """Verify all expected prediction artifacts without rerunning inference."""

    experiment_path = run_dir / "experiment.json"
    if not experiment_path.is_file():
        raise ValueError("run directory is missing experiment.json")
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if experiment.get("status") != "complete":
        raise ValueError("run is not complete")
    fingerprint = str(experiment.get("experiment_fingerprint", ""))
    if not fingerprint:
        raise ValueError("run has no experiment fingerprint")
    samples = select_predictor_samples(
        read_manifest(data_settings.manifest_path),
        data_settings,
    )
    samples = [sample for sample in samples if sample.split in settings.evaluation_splits]
    errors: list[str] = []
    counts: Counter[str] = Counter()
    for sample in samples:
        for method in settings.baselines:
            path = run_dir / "predictions" / method / sample.split / f"{sample.sample_id}.npz"
            _validate_one(path, sample.sample_id, method, fingerprint, errors)
            counts[method] += int(path.is_file())
        for seed in settings.seeds:
            method = f"audio_gru_seed_{seed}"
            path = (
                run_dir / f"seed_{seed}" / "predictions" / sample.split / f"{sample.sample_id}.npz"
            )
            _validate_one(path, sample.sample_id, "audio_gru", fingerprint, errors)
            counts[method] += int(path.is_file())
    report = {
        "sample_count": len(samples),
        "prediction_counts": dict(counts),
        "error_count": len(errors),
        "errors": errors,
        "experiment_fingerprint": fingerprint,
    }
    atomic_write_json(run_dir / "validation_report.json", report)
    return report


def _validate_one(
    path: Path,
    sample_id: str,
    method: str,
    fingerprint: str,
    errors: list[str],
) -> None:
    try:
        payload = load_prediction(path, expected_fingerprint=fingerprint)
        if payload["sample_id"] != sample_id:
            raise ValueError("sample_id does not match")
        if payload["method"] != method:
            raise ValueError("method does not match")
        if payload["prediction"].shape[0] != 75:
            raise ValueError("prediction must have 75 frames")
        if not (payload["prediction"][0] == 0).all():
            raise ValueError("first-frame prediction is not zero")
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"{path}: {exc}")
