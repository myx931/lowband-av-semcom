from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from av_semcom.data.preprocessing import atomic_write_json
from av_semcom.models.full_motion.comparison import run_matched_comparison
from av_semcom.models.full_motion.config import full_motion_jscc_settings
from av_semcom.models.full_motion.experiment import (
    run_full_motion_evaluation,
    run_full_motion_training,
)
from av_semcom.models.full_motion.export import export_full_motion_candidates
from av_semcom.models.jscc.candidates import load_candidate_bundle
from av_semcom.models.jscc.config import JSCCReconstructionSettings
from av_semcom.models.jscc.data import ResidualExample, save_residual_example
from av_semcom.models.motion.perturbations import (
    MotionNormalizer,
    save_motion_normalizer,
)

pytestmark = pytest.mark.smoke


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "channel": {
            "backend": "native_reference",
            "complex_channel_uses": [1],
            "target_power": 1.0,
        },
        "jscc_model": {"input_dim": 18, "hidden_dim": 8},
        "jscc_training": {
            "seeds": [42],
            "device": "cpu",
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "max_epochs": 2,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 0.0,
            "gradient_clip_norm": 1.0,
            "num_workers": 0,
            "deterministic": True,
            "snr_min_db": 0.0,
            "snr_max_db": 5.0,
        },
        "jscc_evaluation": {
            "output_dir": str(tmp_path / "residual"),
            "validation_snr_db": [2.5],
            "test_snr_db": [0.0],
            "noise_seeds": [42],
        },
        "full_motion_jscc": {
            "output_dir": str(tmp_path / "full"),
            "representation": "train_standardized_full_18d_motion",
        },
        "jscc_reconstruction": {
            "split": "test",
            "noise_seed": 42,
            "metric_workers": 1,
            "media_channel_uses": 1,
            "save_representative_media": False,
        },
    }


def _example(split: str, speaker: str, index: int) -> ResidualExample:
    generator = np.random.default_rng(index)
    target = generator.normal(0, 0.1, size=(6, 18)).astype(np.float32)
    prediction = target * np.float32(0.4)
    target[0] = 0
    prediction[0] = 0
    valid = np.ones(6, dtype=np.bool_)
    transmission = valid.copy()
    transmission[0] = False
    raw = target - prediction
    normalized = raw / np.float32(0.1)
    return ResidualExample(
        sample_id=f"{speaker}_{index}",
        speaker_id=speaker,
        split=split,
        prediction=prediction,
        target=target,
        raw_residual=raw,
        normalized_residual=normalized,
        valid_mask=valid,
        transmission_mask=transmission,
    )


def _source_run(tmp_path: Path, stats_path: Path) -> Path:
    run = tmp_path / "e5"
    fingerprint = "e5-smoke"
    examples = [
        *(_example("train", "s1", index) for index in range(4)),
        *(_example("validation", "s2", index + 10) for index in range(2)),
        *(_example("test", "s3", index + 20) for index in range(2)),
    ]
    for example in examples:
        save_residual_example(
            run / "residual_data" / example.split / f"{example.sample_id}.npz",
            example,
            experiment_fingerprint=fingerprint,
        )
    atomic_write_json(
        run / "run_metadata.json",
        {"experiment_fingerprint": fingerprint},
    )
    for filename in ("training_complete.json", "evaluation_complete.json"):
        atomic_write_json(
            run / filename,
            {"experiment_fingerprint": fingerprint, "status": "complete"},
        )
    save_motion_normalizer(
        stats_path,
        MotionNormalizer(
            mean=np.zeros(18, dtype=np.float32),
            std=np.full(18, 0.1, dtype=np.float32),
            scope="train_stats",
        ),
    )
    return run


def test_full_motion_train_evaluate_and_resume(tmp_path: Path) -> None:
    settings = full_motion_jscc_settings(_config(tmp_path))
    stats_path = tmp_path / "motion_stats.json"
    e5 = _source_run(tmp_path, stats_path)
    run = tmp_path / "run"

    output, training = run_full_motion_training(
        settings,
        stats_path,
        e5,
        run_directory=run,
        formal=False,
    )
    summary = run_full_motion_evaluation(
        settings,
        stats_path,
        e5,
        output,
        formal=False,
    )
    reconstruction = JSCCReconstructionSettings.from_config(
        _config(tmp_path),
        settings,
    )
    exported = export_full_motion_candidates(
        settings,
        reconstruction,
        stats_path,
        e5,
        output,
        formal=False,
    )
    bundle = load_candidate_bundle(
        output / "reconstruction_candidates/test/s3_20.npz",
        expected_fingerprint=exported["candidate_fingerprint"],
    )
    full_rows = [
        json.loads(line)
        for line in (output / "test_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    residual_rows = []
    for row in full_rows:
        if row["condition"] != "jscc_awgn":
            continue
        copied = dict(row)
        copied["l1"] = float(copied["l1"]) * 0.9
        copied["rmse"] = float(copied["rmse"]) * 0.9
        copied["velocity_l1"] = float(copied["velocity_l1"]) * 0.9
        residual_rows.append(copied)
    (e5 / "test_metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in residual_rows),
        encoding="utf-8",
    )
    atomic_write_json(
        e5 / "training_summary.json",
        {
            "models": [
                {
                    "channel_uses": 1,
                    "seed": 42,
                    "best_validation_normalized_mse": 0.5,
                }
            ]
        },
    )
    video_common = {
        "family": "jscc_awgn",
        "channel_uses": 1,
        "snr_db": 0.0,
        "noise_seed": 42,
        "sample_count": 2,
        "oracle_psnr_db": 20.0,
        "oracle_ssim": 0.8,
        "oracle_landmark_coverage": 1.0,
    }
    atomic_write_json(
        e5 / "video_reconstruction/summary.json",
        {
            "failure_count": 0,
            "groups": [
                {
                    **video_common,
                    "oracle_mouth_mae": 0.1,
                    "oracle_mouth_nme": 0.1,
                }
            ],
        },
    )
    atomic_write_json(
        output / "video_reconstruction/summary.json",
        {
            "failure_count": 0,
            "groups": [
                {
                    **video_common,
                    "oracle_mouth_mae": 0.2,
                    "oracle_mouth_nme": 0.2,
                }
            ],
        },
    )
    for root in (e5, output):
        atomic_write_json(
            root / "video_reconstruction/complete.json",
            {"status": "complete", "failure_count": 0},
        )
    comparison_dir = tmp_path / "comparison"
    _, comparison = run_matched_comparison(
        tmp_path / "comparison-output",
        e5,
        output,
        run_directory=comparison_dir,
    )
    marker_mtime = (run / "evaluation_complete.json").stat().st_mtime_ns
    resumed_output, resumed_training = run_full_motion_training(
        settings,
        stats_path,
        e5,
        run_directory=run,
        resume=True,
        formal=False,
    )
    resumed_summary = run_full_motion_evaluation(
        settings,
        stats_path,
        e5,
        output,
        resume=True,
        formal=False,
    )

    assert resumed_output == output
    assert resumed_training == training
    assert resumed_summary == summary
    assert training["representation"] == "train_standardized_full_18d_motion"
    assert summary["result_count"] == 8
    assert summary["bitrate_claimed"] is False
    assert exported["condition_count_per_sample"] == 4
    assert [condition.family for condition in bundle.conditions] == [
        "audio_prediction",
        "full_motion_oracle",
        "noiseless_autoencoder",
        "jscc_awgn",
    ]
    assert comparison["motion_pair_count"] == 2
    assert comparison["motion_group_count"] == 1
    assert comparison["video_group_count"] == 1
    assert comparison["residual_lower_l1_group_count"] == 1
    assert (run / "evaluation_complete.json").stat().st_mtime_ns == marker_mtime
