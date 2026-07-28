from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from av_semcom.channel.awgn import NativeComplexAWGN
from av_semcom.data.preprocessing import atomic_write_json
from av_semcom.models.jscc.config import JSCCSettings
from av_semcom.models.jscc.data import ResidualExample, save_residual_example
from av_semcom.models.jscc.experiment import _derived_noise_seed, _metric_row
from av_semcom.models.jscc.model import ResidualJSCC
from av_semcom.models.motion.perturbations import (
    MotionNormalizer,
    save_motion_normalizer,
)
from av_semcom.models.predictor.artifacts import atomic_save_checkpoint
from av_semcom.models.selection.config import (
    ResidualScorerAblationSettings,
    ResidualScorerSettings,
)
from av_semcom.models.selection.gate import GatePolicy
from av_semcom.models.selection.scorer_ablation import (
    run_scorer_ablation_evaluation,
    run_scorer_ablation_training,
)
from av_semcom.models.selection.scorer_experiment import (
    run_scorer_evaluation,
    run_scorer_training,
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
            "snr_min_db": 0.0,
            "snr_max_db": 5.0,
        },
        "jscc_evaluation": {
            "output_dir": str(tmp_path / "e5-output"),
            "validation_snr_db": [1.5],
            "test_snr_db": [0.0],
            "noise_seeds": [42],
        },
        "residual_scorer": {
            "output_dir": str(tmp_path / "scorer-output"),
            "budgets_by_channel_use": {1: 2},
            "hidden_dim": 8,
            "temperature": 1.0,
            "velocity_weight": 0.5,
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
            "validation_snr_db": [2.5],
            "noise_seeds": [42],
            "random_seeds": [42],
        },
        "residual_scorer_ablation": {
            "output_dir": str(tmp_path / "scorer-ablation-output"),
            "channel_uses": [1],
            "calibration_sample_count": 1,
            "partition_salt": "smoke",
            "variants": {
                "full": {"use_snr": True, "velocity_weight": 0.5},
                "no_snr": {"use_snr": False, "velocity_weight": 0.5},
                "no_velocity": {"use_snr": True, "velocity_weight": 0.0},
                "no_snr_no_velocity": {
                    "use_snr": False,
                    "velocity_weight": 0.0,
                },
            },
        },
    }


def _example(split: str, speaker: str, index: int) -> ResidualExample:
    generator = np.random.default_rng(index)
    normalized = generator.normal(size=(6, 18)).astype(np.float32)
    normalized[0] = 0
    valid = np.ones(6, dtype=np.bool_)
    transmission = valid.copy()
    transmission[0] = False
    raw = normalized * np.float32(0.1)
    prediction = np.zeros_like(raw)
    return ResidualExample(
        sample_id=f"{speaker}_{index}",
        speaker_id=speaker,
        split=split,
        prediction=prediction,
        target=raw.copy(),
        raw_residual=raw,
        normalized_residual=normalized,
        valid_mask=valid,
        transmission_mask=transmission,
    )


def test_residual_scorer_train_evaluate_and_resume(tmp_path: Path) -> None:
    config = _config(tmp_path)
    jscc = JSCCSettings.from_config(config)
    scorer_settings = ResidualScorerSettings.from_config(config, jscc)
    e5 = tmp_path / "e5"
    gate = tmp_path / "gate"
    run_dir = tmp_path / "scorer-run"
    fingerprint = "e5-smoke"
    examples = [
        *(_example("train", "s1", index) for index in range(4)),
        *(_example("validation", "s2", index + 10) for index in range(2)),
        *(_example("test", "s3", index + 20) for index in range(2)),
    ]
    for example in examples:
        save_residual_example(
            e5 / "residual_data" / example.split / f"{example.sample_id}.npz",
            example,
            experiment_fingerprint=fingerprint,
        )
    atomic_write_json(
        e5 / "residual_data/train_validation_complete.json",
        {"experiment_fingerprint": fingerprint, "sample_count": 6},
    )
    atomic_write_json(
        e5 / "run_metadata.json",
        {"experiment_fingerprint": fingerprint},
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
    e5_model = ResidualJSCC(
        channel=NativeComplexAWGN(seed=42),
        input_dim=18,
        hidden_dim=8,
        channel_uses=1,
    )
    atomic_save_checkpoint(
        e5 / "models/c_1/seed_42/best.pt",
        {
            "experiment_fingerprint": fingerprint,
            "model_state": e5_model.state_dict(),
        },
    )
    policy = GatePolicy(
        experiment_fingerprint=fingerprint,
        gate_fingerprint="gate-smoke",
        selected_model_seeds={1: 42},
        thresholds_db={1: -1.0},
        calibration_snr_db=(-0.5,),
        primary_metric="l1",
        minimum_relative_improvement=0.0,
    )
    atomic_write_json(gate / "policy.json", policy.to_dict())
    atomic_write_json(
        gate / "complete.json",
        {
            "gate_fingerprint": policy.gate_fingerprint,
            "status": "complete",
        },
    )
    std = np.full(18, 0.1, dtype=np.float32)
    stats_path = tmp_path / "train_stats.json"
    save_motion_normalizer(
        stats_path,
        MotionNormalizer(
            mean=np.zeros(18, dtype=np.float32),
            std=std,
            scope="train_stats",
        ),
    )
    predictor = SimpleNamespace(motion_stats_path=stats_path)
    atomic_write_json(
        e5 / "training_complete.json",
        {
            "experiment_fingerprint": fingerprint,
            "status": "complete",
        },
    )

    source_rows = []
    e5_model.eval()
    for example_index, example in enumerate(
        example for example in examples if example.split == "test"
    ):
        residual = torch.from_numpy(example.normalized_residual).unsqueeze(0)
        mask = torch.from_numpy(example.transmission_mask).unsqueeze(0)
        decoded = (
            e5_model(
                residual,
                mask,
                0.0,
                noise_seed=_derived_noise_seed(42, 42, 0, example_index),
            )
            .decoded_residual[0]
            .detach()
            .numpy()
            .astype(np.float32)
        )
        source_rows.append(
            _metric_row(
                example,
                decoded,
                std,
                condition="jscc_awgn",
                channel_uses=1,
                model_seed=42,
                snr_db=0.0,
                noise_seed=42,
            )
        )
    (e5 / "test_metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in source_rows),
        encoding="utf-8",
    )

    output, training = run_scorer_training(
        scorer_settings,
        jscc,
        predictor,  # type: ignore[arg-type]
        e5,
        gate,
        run_directory=run_dir,
        formal=False,
    )
    summary = run_scorer_evaluation(
        scorer_settings,
        jscc,
        predictor,  # type: ignore[arg-type]
        e5,
        gate,
        run_dir,
        formal=False,
    )
    mtime = (run_dir / "evaluation_complete.json").stat().st_mtime_ns
    resumed = run_scorer_evaluation(
        scorer_settings,
        jscc,
        predictor,  # type: ignore[arg-type]
        e5,
        gate,
        run_dir,
        resume=True,
        formal=False,
    )

    assert output == run_dir
    assert len(training["models"]) == 1
    assert summary == resumed
    assert summary["result_count"] == 12
    assert summary["maximum_dense_metric_difference"] == pytest.approx(0.0)
    assert (run_dir / "evaluation_complete.json").stat().st_mtime_ns == mtime

    (e5 / "test_metrics.jsonl").unlink()
    ablation = ResidualScorerAblationSettings.from_config(
        config,
        jscc,
        scorer_settings,
    )
    ablation_dir = tmp_path / "scorer-ablation-run"
    ablation_output, ablation_training = run_scorer_ablation_training(
        ablation,
        scorer_settings,
        jscc,
        predictor,  # type: ignore[arg-type]
        e5,
        run_directory=ablation_dir,
        formal=False,
    )
    ablation_summary = run_scorer_ablation_evaluation(
        ablation,
        scorer_settings,
        jscc,
        predictor,  # type: ignore[arg-type]
        e5,
        ablation_dir,
        formal=False,
    )
    ablation_mtime = (ablation_dir / "audit_complete.json").stat().st_mtime_ns
    _, ablation_training_resumed = run_scorer_ablation_training(
        ablation,
        scorer_settings,
        jscc,
        predictor,  # type: ignore[arg-type]
        e5,
        run_directory=ablation_dir,
        resume=True,
        formal=False,
    )
    ablation_summary_resumed = run_scorer_ablation_evaluation(
        ablation,
        scorer_settings,
        jscc,
        predictor,  # type: ignore[arg-type]
        e5,
        ablation_dir,
        resume=True,
        formal=False,
    )

    assert ablation_output == ablation_dir
    assert ablation_training["model_count"] == 4
    assert ablation_training_resumed == ablation_training
    assert ablation_training["test_data_accessed"] is False
    assert ablation_summary["result_count"] == 6
    assert ablation_summary_resumed == ablation_summary
    assert ablation_summary["test_data_accessed"] is False
    assert (ablation_dir / "audit_complete.json").stat().st_mtime_ns == ablation_mtime
    assert not (e5 / "test_metrics.jsonl").exists()
