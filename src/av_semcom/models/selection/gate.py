"""Validation-calibrated hard SNR gate for the frozen E5 residual JSCC."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from av_semcom.data.preprocessing import atomic_write_json, config_fingerprint
from av_semcom.metrics.motion import compute_motion_metrics
from av_semcom.models.jscc.config import JSCCSettings
from av_semcom.models.jscc.data import ResidualExample, load_residual_example
from av_semcom.models.jscc.experiment import (
    _build_model,
    _derived_noise_seed,
    _metric_row,
)
from av_semcom.models.jscc.export import select_validation_model_seeds
from av_semcom.models.motion.perturbations import load_motion_normalizer
from av_semcom.models.predictor.artifacts import file_sha256, load_checkpoint
from av_semcom.models.predictor.config import AudioMotionSettings
from av_semcom.models.selection.config import ChannelGateSettings

_METRICS = ("normalized_residual_mse", "l1", "rmse", "velocity_l1")


@dataclass(frozen=True)
class GatePolicy:
    """A validation-frozen global SNR threshold for every channel budget."""

    experiment_fingerprint: str
    gate_fingerprint: str
    selected_model_seeds: Mapping[int, int]
    thresholds_db: Mapping[int, float | None]
    calibration_snr_db: tuple[float, ...]
    primary_metric: str
    minimum_relative_improvement: float

    def should_transmit(self, channel_uses: int, snr_db: float) -> bool:
        """Return the deployment decision without consulting test metrics."""

        if channel_uses not in self.thresholds_db:
            raise ValueError(f"gate policy has no threshold for C={channel_uses}")
        threshold = self.thresholds_db[channel_uses]
        return threshold is not None and float(snr_db) >= threshold

    def to_dict(self) -> dict[str, Any]:
        """Serialize with string keys for portable JSON."""

        return {
            "schema_version": 1,
            "experiment_fingerprint": self.experiment_fingerprint,
            "gate_fingerprint": self.gate_fingerprint,
            "selection_rule": (
                "minimum validation SNR whose complete higher-SNR suffix "
                "strictly improves validation mean L1"
            ),
            "selected_model_seeds": {
                str(key): value for key, value in self.selected_model_seeds.items()
            },
            "thresholds_db": {str(key): value for key, value in self.thresholds_db.items()},
            "calibration_snr_db": list(self.calibration_snr_db),
            "primary_metric": self.primary_metric,
            "minimum_relative_improvement": self.minimum_relative_improvement,
            "below_calibration_grid_action": "prediction_only",
            "no_safe_suffix_action": "prediction_only",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GatePolicy:
        """Load a policy while rejecting incomplete mappings."""

        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError("unsupported gate policy schema")
        seeds = payload.get("selected_model_seeds")
        thresholds = payload.get("thresholds_db")
        calibration = payload.get("calibration_snr_db")
        if not isinstance(seeds, Mapping) or not isinstance(thresholds, Mapping):
            raise ValueError("gate policy is missing seed or threshold mappings")
        if not isinstance(calibration, list) or not calibration:
            raise ValueError("gate policy is missing its calibration grid")
        parsed_thresholds = {
            int(key): None if value is None else float(value) for key, value in thresholds.items()
        }
        return cls(
            experiment_fingerprint=str(payload["experiment_fingerprint"]),
            gate_fingerprint=str(payload["gate_fingerprint"]),
            selected_model_seeds={int(key): int(value) for key, value in seeds.items()},
            thresholds_db=parsed_thresholds,
            calibration_snr_db=tuple(float(value) for value in calibration),
            primary_metric=str(payload["primary_metric"]),
            minimum_relative_improvement=float(payload["minimum_relative_improvement"]),
        )


def derive_gate_policy(
    rows: Sequence[Mapping[str, Any]],
    settings: ChannelGateSettings,
    jscc: JSCCSettings,
    selected_model_seeds: Mapping[int, int],
    *,
    experiment_fingerprint: str,
    gate_fingerprint: str,
) -> tuple[GatePolicy, dict[str, Any]]:
    """Fit one monotonic safety threshold per C using validation rows only."""

    if not rows or any(row.get("split") != "validation" for row in rows):
        raise ValueError("gate calibration accepts validation rows only")
    prediction_rows = [row for row in rows if row.get("condition") == "prediction_only"]
    if not prediction_rows:
        raise ValueError("gate calibration has no prediction-only baseline")
    sample_ids = {str(row["sample_id"]) for row in prediction_rows}
    if len(prediction_rows) != len(sample_ids):
        raise ValueError("gate calibration has duplicate prediction-only rows")
    prediction_metric = float(
        np.mean([float(row[settings.primary_metric]) for row in prediction_rows])
    )
    target_metric = prediction_metric * (1.0 - settings.minimum_relative_improvement)

    thresholds: dict[int, float | None] = {}
    calibration_groups: list[dict[str, Any]] = []
    expected_count = len(sample_ids) * len(settings.noise_seeds)
    for channel_uses in jscc.channel_uses:
        seed = selected_model_seeds[channel_uses]
        points: list[dict[str, Any]] = []
        for snr_db in settings.validation_snr_db:
            members = [
                row
                for row in rows
                if row.get("condition") == "jscc_awgn"
                and int(row["channel_uses"]) == channel_uses
                and int(row["model_seed"]) == seed
                and float(row["snr_db"]) == snr_db
            ]
            if len(members) != expected_count:
                raise ValueError(
                    f"expected {expected_count} validation rows for "
                    f"C={channel_uses}, SNR={snr_db}; found {len(members)}"
                )
            if {int(row["noise_seed"]) for row in members} != set(settings.noise_seeds):
                raise ValueError("gate calibration noise seed set is incomplete")
            metric = float(np.mean([float(row[settings.primary_metric]) for row in members]))
            points.append(
                {
                    "channel_uses": channel_uses,
                    "model_seed": seed,
                    "snr_db": snr_db,
                    "sample_noise_realization_count": len(members),
                    "prediction_l1": prediction_metric,
                    "jscc_l1": metric,
                    "relative_improvement": 1.0 - metric / prediction_metric,
                    "point_improves_prediction": metric < target_metric,
                }
            )

        threshold: float | None = None
        for index, point in enumerate(points):
            if all(bool(candidate["point_improves_prediction"]) for candidate in points[index:]):
                threshold = float(point["snr_db"])
                break
        thresholds[channel_uses] = threshold
        for point in points:
            point["selected_threshold_db"] = threshold
            point["gate_transmits"] = threshold is not None and float(point["snr_db"]) >= threshold
            calibration_groups.append(point)

    policy = GatePolicy(
        experiment_fingerprint=experiment_fingerprint,
        gate_fingerprint=gate_fingerprint,
        selected_model_seeds=dict(selected_model_seeds),
        thresholds_db=thresholds,
        calibration_snr_db=settings.validation_snr_db,
        primary_metric=settings.primary_metric,
        minimum_relative_improvement=settings.minimum_relative_improvement,
    )
    summary = {
        "schema_version": 1,
        "split": "validation",
        "selection_used_test_metrics": False,
        "primary_metric": settings.primary_metric,
        "minimum_relative_improvement": settings.minimum_relative_improvement,
        "prediction_l1": prediction_metric,
        "sample_count": len(sample_ids),
        "result_count": len(rows),
        "groups": calibration_groups,
        "policy": policy.to_dict(),
    }
    return policy, summary


def apply_frozen_gate_to_test(
    source_rows: Sequence[Mapping[str, Any]],
    policy: GatePolicy,
    jscc: JSCCSettings,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply a frozen policy to immutable E5 test rows without tuning it."""

    if not source_rows or any(row.get("split") != "test" for row in source_rows):
        raise ValueError("frozen gate evaluation accepts test rows only")
    prediction_rows = [row for row in source_rows if row.get("condition") == "prediction_only"]
    prediction_index = {str(row["sample_id"]): row for row in prediction_rows}
    if len(prediction_index) != len(prediction_rows) or not prediction_index:
        raise ValueError("test rows have missing or duplicate prediction-only baselines")
    prediction_means = {
        metric: float(np.mean([float(row[metric]) for row in prediction_rows]))
        for metric in _METRICS
    }

    output: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    expected_count = len(prediction_index) * len(jscc.noise_seeds)
    for channel_uses in jscc.channel_uses:
        model_seed = policy.selected_model_seeds[channel_uses]
        for snr_db in jscc.test_snr_db:
            always_rows = [
                row
                for row in source_rows
                if row.get("condition") == "jscc_awgn"
                and int(row["channel_uses"]) == channel_uses
                and int(row["model_seed"]) == model_seed
                and float(row["snr_db"]) == snr_db
            ]
            if len(always_rows) != expected_count:
                raise ValueError(
                    f"expected {expected_count} frozen test rows for "
                    f"C={channel_uses}, SNR={snr_db}; found {len(always_rows)}"
                )
            if {int(row["noise_seed"]) for row in always_rows} != set(jscc.noise_seeds):
                raise ValueError("frozen test noise seed set is incomplete")
            transmit = policy.should_transmit(channel_uses, snr_db)
            gated_members: list[dict[str, Any]] = []
            for row in always_rows:
                baseline = prediction_index[str(row["sample_id"])]
                chosen = row if transmit else baseline
                gated = {
                    "sample_id": row["sample_id"],
                    "speaker_id": row["speaker_id"],
                    "split": "test",
                    "condition": "validation_snr_gate",
                    "source_condition": "jscc_awgn" if transmit else "prediction_only",
                    "channel_uses": channel_uses,
                    "model_seed": model_seed,
                    "snr_db": snr_db,
                    "noise_seed": int(row["noise_seed"]),
                    "threshold_db": policy.thresholds_db[channel_uses],
                    "transmit": transmit,
                    "complex_channel_uses_used": channel_uses if transmit else 0,
                    **{metric: float(chosen[metric]) for metric in _METRICS},
                }
                output.append(gated)
                gated_members.append(gated)

            always_means = {
                metric: float(np.mean([float(row[metric]) for row in always_rows]))
                for metric in _METRICS
            }
            gated_means = {
                metric: float(np.mean([float(row[metric]) for row in gated_members]))
                for metric in _METRICS
            }
            groups.append(
                {
                    "channel_uses": channel_uses,
                    "model_seed": model_seed,
                    "snr_db": snr_db,
                    "threshold_db": policy.thresholds_db[channel_uses],
                    "decision": "send_jscc" if transmit else "prediction_only",
                    "transmit_fraction": 1.0 if transmit else 0.0,
                    "mean_complex_channel_uses": float(channel_uses if transmit else 0),
                    "sample_noise_realization_count": len(gated_members),
                    **{f"prediction_{metric}": value for metric, value in prediction_means.items()},
                    **{f"always_send_{metric}": value for metric, value in always_means.items()},
                    **{f"gated_{metric}": value for metric, value in gated_means.items()},
                    "gated_relative_l1_improvement_vs_prediction": (
                        1.0 - gated_means["l1"] / prediction_means["l1"]
                    ),
                    "always_send_relative_l1_improvement_vs_prediction": (
                        1.0 - always_means["l1"] / prediction_means["l1"]
                    ),
                }
            )
    return output, {
        "schema_version": 1,
        "status": "complete",
        "split": "test",
        "policy_selected_on": "validation_only",
        "test_used_for_policy_selection": False,
        "sample_count": len(prediction_index),
        "result_count": len(output),
        "groups": groups,
        "policy": policy.to_dict(),
    }


@torch.no_grad()
def evaluate_validation_gate_grid(
    examples: Sequence[ResidualExample],
    models: Mapping[int, torch.nn.Module],
    selected_model_seeds: Mapping[int, int],
    settings: ChannelGateSettings,
    motion_std: np.ndarray,
) -> list[dict[str, Any]]:
    """Evaluate frozen selected models on the disjoint validation SNR grid."""

    if not examples or any(example.split != "validation" for example in examples):
        raise ValueError("gate grid evaluation requires validation examples only")
    rows = [_prediction_metric_row(example) for example in examples]
    std = np.asarray(motion_std, dtype=np.float32)
    for channel_uses, model in models.items():
        model.eval()
        model_seed = selected_model_seeds[channel_uses]
        device = next(model.parameters()).device
        for snr_index, snr_db in enumerate(settings.validation_snr_db):
            for noise_seed in settings.noise_seeds:
                for example_index, example in enumerate(examples):
                    residual = torch.from_numpy(example.normalized_residual).unsqueeze(0).to(device)
                    mask = torch.from_numpy(example.transmission_mask).unsqueeze(0).to(device)
                    result = model(
                        residual,
                        mask,
                        snr_db,
                        noise_seed=_derived_noise_seed(
                            model_seed,
                            noise_seed,
                            snr_index,
                            example_index,
                        ),
                    )
                    decoded = result.decoded_residual[0].cpu().numpy().astype(np.float32)
                    rows.append(
                        _metric_row(
                            example,
                            decoded,
                            std,
                            condition="jscc_awgn",
                            channel_uses=channel_uses,
                            model_seed=model_seed,
                            snr_db=snr_db,
                            noise_seed=noise_seed,
                        )
                    )
    return rows


def run_channel_gate_experiment(
    settings: ChannelGateSettings,
    jscc: JSCCSettings,
    predictor: AudioMotionSettings | None,
    e5_run_dir: Path,
    *,
    run_directory: Path | None = None,
    resume: bool = False,
    formal: bool = True,
    calibration_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Calibrate on validation, freeze the policy, then evaluate test once."""

    if formal:
        jscc.require_formal_backend()
    e5_run_dir = e5_run_dir.resolve()
    source = _source_provenance(e5_run_dir, jscc)
    fingerprint = config_fingerprint({"channel_gate": settings.config, "source": source})
    run_dir = (
        run_directory.resolve()
        if run_directory is not None
        else _new_run_directory(settings.output_root)
    )
    _prepare_run_directory(
        run_dir,
        settings,
        source,
        fingerprint,
        resume=resume,
    )
    complete_path = run_dir / "complete.json"
    if complete_path.is_file():
        if not resume:
            raise FileExistsError("channel gate run is complete; pass --resume")
        complete = _read_json(complete_path)
        _require_gate_fingerprint(complete, fingerprint, complete_path)
        source_test_hash = file_sha256(e5_run_dir / "test_metrics.jsonl")
        if complete.get("source_test_metrics_sha256") != source_test_hash:
            raise ValueError("source E5 test metrics changed after gate evaluation")
        expected_evaluation_fingerprint = config_fingerprint(
            {
                "gate_fingerprint": fingerprint,
                "source_test_metrics_sha256": source_test_hash,
            }
        )
        if complete.get("evaluation_fingerprint") != expected_evaluation_fingerprint:
            raise ValueError("channel gate test evaluation fingerprint mismatch")
        return run_dir, _read_json(run_dir / "test_summary.json")

    training_summary = _read_json(e5_run_dir / "training_summary.json")
    selected_seeds = select_validation_model_seeds(
        training_summary,
        jscc.channel_uses,
    )
    calibration_marker = run_dir / "calibration_complete.json"
    if calibration_marker.is_file():
        if not resume:
            raise FileExistsError("gate calibration exists; pass --resume")
        _require_gate_fingerprint(
            _read_json(calibration_marker),
            fingerprint,
            calibration_marker,
        )
        policy = GatePolicy.from_dict(_read_json(run_dir / "policy.json"))
    else:
        if calibration_rows is None:
            if predictor is None:
                raise ValueError("predictor settings are required for live calibration")
            examples = _load_validation_examples(
                e5_run_dir,
                str(source["experiment_fingerprint"]),
            )
            device = _resolve_device(jscc.device)
            models = {
                channel_uses: _load_model(
                    jscc,
                    e5_run_dir,
                    str(source["experiment_fingerprint"]),
                    channel_uses,
                    selected_seeds[channel_uses],
                    device,
                )
                for channel_uses in jscc.channel_uses
            }
            normalizer = load_motion_normalizer(predictor.motion_stats_path)
            if normalizer.scope != "train_stats":
                raise ValueError("channel gate requires train-only motion statistics")
            active_rows = evaluate_validation_gate_grid(
                examples,
                models,
                selected_seeds,
                settings,
                normalizer.std,
            )
        else:
            active_rows = [dict(row) for row in calibration_rows]
        policy, calibration_summary = derive_gate_policy(
            active_rows,
            settings,
            jscc,
            selected_seeds,
            experiment_fingerprint=str(source["experiment_fingerprint"]),
            gate_fingerprint=fingerprint,
        )
        _atomic_write_jsonl(run_dir / "validation_metrics.jsonl", active_rows)
        atomic_write_json(run_dir / "validation_summary.json", calibration_summary)
        atomic_write_json(run_dir / "policy.json", policy.to_dict())
        _write_dict_csv(
            run_dir / "validation_summary.csv",
            calibration_summary["groups"],
        )
        _write_calibration_plot(run_dir / "plots", calibration_summary)
        atomic_write_json(
            calibration_marker,
            {
                "gate_fingerprint": fingerprint,
                "result_count": len(active_rows),
                "policy_sha256": file_sha256(run_dir / "policy.json"),
                "status": "complete",
            },
        )

    if policy.gate_fingerprint != fingerprint:
        raise ValueError("frozen gate policy fingerprint does not match")
    source_test_path = e5_run_dir / "test_metrics.jsonl"
    source_test_hash = file_sha256(source_test_path)
    evaluation_fingerprint = config_fingerprint(
        {
            "gate_fingerprint": fingerprint,
            "source_test_metrics_sha256": source_test_hash,
        }
    )
    atomic_write_json(
        run_dir / "test_provenance.json",
        {
            "evaluation_fingerprint": evaluation_fingerprint,
            "gate_fingerprint": fingerprint,
            "source_test_metrics_sha256": source_test_hash,
        },
    )
    source_test_rows = _read_jsonl(source_test_path)
    gated_rows, test_summary = apply_frozen_gate_to_test(
        source_test_rows,
        policy,
        jscc,
    )
    test_summary.update(
        {
            "evaluation_fingerprint": evaluation_fingerprint,
            "gate_fingerprint": fingerprint,
            "source_test_metrics_sha256": source_test_hash,
        }
    )
    _atomic_write_jsonl(run_dir / "test_metrics.jsonl", gated_rows)
    atomic_write_json(run_dir / "test_summary.json", test_summary)
    _write_dict_csv(run_dir / "test_summary.csv", test_summary["groups"])
    _write_test_plot(run_dir / "plots", test_summary)
    atomic_write_json(
        complete_path,
        {
            "evaluation_fingerprint": evaluation_fingerprint,
            "gate_fingerprint": fingerprint,
            "source_test_metrics_sha256": source_test_hash,
            "validation_result_count": int(_read_json(calibration_marker)["result_count"]),
            "test_result_count": len(gated_rows),
            "status": "complete",
        },
    )
    return run_dir, test_summary


def _prediction_metric_row(example: ResidualExample) -> dict[str, Any]:
    metrics = compute_motion_metrics(
        example.target[example.valid_mask],
        example.prediction[example.valid_mask],
    )
    return {
        "sample_id": example.sample_id,
        "speaker_id": example.speaker_id,
        "split": example.split,
        "condition": "prediction_only",
        "channel_uses": None,
        "model_seed": None,
        "snr_db": None,
        "noise_seed": None,
        "normalized_residual_mse": float(
            np.square(
                example.normalized_residual[example.transmission_mask],
                dtype=np.float64,
            ).mean()
        ),
        **metrics.to_dict(),
    }


def _source_provenance(
    e5_run_dir: Path,
    jscc: JSCCSettings,
) -> dict[str, Any]:
    metadata = _read_json(e5_run_dir / "run_metadata.json")
    fingerprint = str(metadata.get("experiment_fingerprint", ""))
    if not fingerprint:
        raise ValueError("E5 run has no experiment fingerprint")
    for marker_name in ("training_complete.json", "evaluation_complete.json"):
        marker = _read_json(e5_run_dir / marker_name)
        if marker.get("experiment_fingerprint") != fingerprint:
            raise ValueError(f"E5 completion fingerprint mismatch: {marker_name}")
    training_summary_path = e5_run_dir / "training_summary.json"
    selected = select_validation_model_seeds(
        _read_json(training_summary_path),
        jscc.channel_uses,
    )
    checkpoints = {
        str(channel_uses): file_sha256(
            e5_run_dir
            / "models"
            / f"c_{channel_uses}"
            / f"seed_{selected[channel_uses]}"
            / "best.pt"
        )
        for channel_uses in jscc.channel_uses
    }
    validation_marker = e5_run_dir / "residual_data/train_validation_complete.json"
    return {
        "experiment_fingerprint": fingerprint,
        "channel_backend": jscc.channel_backend,
        "training_summary_sha256": file_sha256(training_summary_path),
        "validation_cache_marker_sha256": file_sha256(validation_marker),
        "selected_model_seeds": {str(key): value for key, value in selected.items()},
        "selected_checkpoint_sha256": checkpoints,
        "selection_rule": "minimum_validation_normalized_mse_per_channel_use",
    }


def _load_validation_examples(
    e5_run_dir: Path,
    experiment_fingerprint: str,
) -> list[ResidualExample]:
    paths = sorted((e5_run_dir / "residual_data/validation").glob("*.npz"))
    if not paths:
        raise ValueError("E5 run has no cached validation residuals")
    examples = [
        load_residual_example(path, expected_fingerprint=experiment_fingerprint) for path in paths
    ]
    if any(example.split != "validation" for example in examples):
        raise ValueError("E5 validation cache contains another split")
    return examples


def _load_model(
    settings: JSCCSettings,
    e5_run_dir: Path,
    experiment_fingerprint: str,
    channel_uses: int,
    model_seed: int,
    device: torch.device,
) -> torch.nn.Module:
    model = _build_model(settings, channel_uses, model_seed, device)
    checkpoint = load_checkpoint(
        e5_run_dir / "models" / f"c_{channel_uses}" / f"seed_{model_seed}" / "best.pt",
        expected_fingerprint=experiment_fingerprint,
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _prepare_run_directory(
    run_dir: Path,
    settings: ChannelGateSettings,
    source: Mapping[str, Any],
    fingerprint: str,
    *,
    resume: bool,
) -> None:
    metadata_path = run_dir / "run_metadata.json"
    if run_dir.exists():
        if not resume:
            raise FileExistsError(f"channel gate run directory exists: {run_dir}")
        _require_gate_fingerprint(
            _read_json(metadata_path),
            fingerprint,
            metadata_path,
        )
        return
    if resume:
        raise FileNotFoundError(f"cannot resume missing channel gate run: {run_dir}")
    run_dir.mkdir(parents=True)
    atomic_write_json(run_dir / "resolved_config.json", dict(settings.config))
    atomic_write_json(run_dir / "source_provenance.json", dict(source))
    atomic_write_json(run_dir / "environment.json", _environment())
    atomic_write_json(
        metadata_path,
        {
            "gate_fingerprint": fingerprint,
            "git_commit": _git_commit(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _require_gate_fingerprint(
    payload: Mapping[str, Any],
    fingerprint: str,
    path: Path,
) -> None:
    if payload.get("gate_fingerprint") != fingerprint:
        raise ValueError(f"channel gate fingerprint mismatch: {path}")


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA gate calibration was requested but is unavailable")
    return device


def _new_run_directory(root: Path) -> Path:
    return root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _environment() -> dict[str, Any]:
    try:
        sionna_version = importlib.metadata.version("sionna-no-rt")
    except importlib.metadata.PackageNotFoundError:
        sionna_version = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "sionna_no_rt": sionna_version,
    }


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.replace(temporary, path)


def _write_dict_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_calibration_plot(
    path: Path,
    summary: Mapping[str, Any],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    groups = summary["groups"]
    if not groups:
        return
    path.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.axhline(
        float(summary["prediction_l1"]),
        color="black",
        linestyle="--",
        label="prediction only",
    )
    for channel_uses in sorted({int(group["channel_uses"]) for group in groups}):
        members = [group for group in groups if int(group["channel_uses"]) == channel_uses]
        axis.plot(
            [float(group["snr_db"]) for group in members],
            [float(group["jscc_l1"]) for group in members],
            marker="o",
            label=f"C={channel_uses}",
        )
    axis.set_xlabel("Validation SNR (dB)")
    axis.set_ylabel("Motion L1")
    axis.set_title("Validation-only SNR gate calibration")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path / "validation_gate_calibration.png", dpi=160)
    plt.close(figure)


def _write_test_plot(
    path: Path,
    summary: Mapping[str, Any],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    groups = summary["groups"]
    if not groups:
        return
    path.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    prediction = float(groups[0]["prediction_l1"])
    axis.axhline(
        prediction,
        color="black",
        linestyle="--",
        label="prediction only",
    )
    for channel_uses in sorted({int(group["channel_uses"]) for group in groups}):
        members = [group for group in groups if int(group["channel_uses"]) == channel_uses]
        axis.plot(
            [float(group["snr_db"]) for group in members],
            [float(group["gated_l1"]) for group in members],
            marker="o",
            label=f"gated C={channel_uses}",
        )
        axis.plot(
            [float(group["snr_db"]) for group in members],
            [float(group["always_send_l1"]) for group in members],
            alpha=0.35,
            linestyle=":",
        )
    axis.set_xlabel("Test SNR (dB)")
    axis.set_ylabel("Motion L1")
    axis.set_title("Validation-frozen SNR gate on test")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path / "test_gate_l1_vs_snr.png", dpi=160)
    plt.close(figure)
