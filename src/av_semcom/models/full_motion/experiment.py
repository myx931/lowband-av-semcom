"""Training and evaluation for the matched full-motion JSCC control."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from av_semcom.data.preprocessing import atomic_write_json, config_fingerprint
from av_semcom.metrics.motion import compute_motion_metrics
from av_semcom.models.full_motion.data import (
    FullMotionData,
    data_audit,
    load_full_motion_data,
)
from av_semcom.models.jscc.config import JSCCSettings
from av_semcom.models.jscc.experiment import (
    _atomic_write_jsonl,
    _build_model,
    _environment,
    _evaluate_one_model,
    _git_commit,
    _new_run_directory,
    _read_json,
    _require_fingerprint,
    _resolve_device,
    _train_one_model,
    _write_csv,
)
from av_semcom.models.motion.perturbations import (
    MotionNormalizer,
    load_motion_normalizer,
)
from av_semcom.models.predictor.artifacts import (
    file_sha256,
    load_checkpoint,
)

_REPRESENTATION = "train_standardized_full_18d_motion"
_METRICS = ("normalized_motion_mse", "l1", "rmse", "velocity_l1")


def run_full_motion_training(
    settings: JSCCSettings,
    motion_stats_path: Path,
    e5_run_dir: Path,
    *,
    run_directory: Path | None = None,
    resume: bool = False,
    formal: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Train only the missing full-motion control on frozen E5 identities."""

    if formal:
        settings.require_formal_backend()
    e5_run_dir = e5_run_dir.resolve()
    inputs = _input_provenance(settings, motion_stats_path, e5_run_dir)
    fingerprint = _fingerprint(settings, inputs)
    run_dir = (
        run_directory.resolve()
        if run_directory is not None
        else _new_run_directory(settings.output_root)
    )
    _prepare_run(run_dir, settings, inputs, fingerprint, resume=resume)
    normalizer = _load_train_normalizer(motion_stats_path)
    data = load_full_motion_data(
        e5_run_dir,
        normalizer,
        splits=("train", "validation"),
    )
    atomic_write_json(run_dir / "data_audit/train_validation.json", data_audit(data))
    train = [item.transport for item in data if item.source.split == "train"]
    validation = [item.transport for item in data if item.source.split == "validation"]
    if formal and (len(train) != 800 or len(validation) != 100):
        raise ValueError(
            f"formal source must contain 800/100 train/validation samples, "
            f"got {len(train)}/{len(validation)}"
        )
    if not train or not validation:
        raise ValueError("training and validation data must be non-empty")

    rows: list[dict[str, Any]] = []
    for channel_uses in settings.channel_uses:
        for seed in settings.seeds:
            model_dir = run_dir / "models" / f"c_{channel_uses}" / f"seed_{seed}"
            marker = model_dir / "complete.json"
            if resume and marker.is_file():
                payload = _read_json(marker)
                _require_fingerprint(payload, fingerprint, marker)
                rows.append(payload)
                continue
            rows.append(
                _train_one_model(
                    settings,
                    train,
                    validation,
                    channel_uses,
                    seed,
                    model_dir,
                    fingerprint,
                    log_label="full-motion-jscc",
                )
            )
    summary = {
        "schema_version": 1,
        "status": "training_complete",
        "experiment_fingerprint": fingerprint,
        "representation": _REPRESENTATION,
        "source_e5_experiment_fingerprint": inputs["e5_experiment_fingerprint"],
        "channel_backend": settings.channel_backend,
        "channel_use_unit": "complex_symbols_per_eligible_frame",
        "bitrate_claimed": False,
        "models": rows,
    }
    atomic_write_json(run_dir / "training_summary.json", summary)
    _write_csv(
        run_dir / "training_summary.csv",
        rows,
        (
            "channel_backend",
            "channel_uses",
            "real_degrees_of_freedom",
            "semantic_compression_ratio",
            "seed",
            "best_epoch",
            "best_validation_normalized_mse",
            "epoch_count",
        ),
    )
    atomic_write_json(
        run_dir / "training_complete.json",
        {
            "experiment_fingerprint": fingerprint,
            "model_count": len(rows),
            "status": "complete",
        },
    )
    return run_dir, summary


def run_full_motion_evaluation(
    settings: JSCCSettings,
    motion_stats_path: Path,
    e5_run_dir: Path,
    run_dir: Path,
    *,
    resume: bool = False,
    formal: bool = True,
) -> dict[str, Any]:
    """Evaluate all full-motion checkpoints once on the frozen E5 test identities."""

    if formal:
        settings.require_formal_backend()
    e5_run_dir = e5_run_dir.resolve()
    run_dir = run_dir.resolve()
    inputs = _input_provenance(settings, motion_stats_path, e5_run_dir)
    expected = _fingerprint(settings, inputs)
    metadata = _read_json(run_dir / "run_metadata.json")
    fingerprint = str(metadata.get("experiment_fingerprint", ""))
    if fingerprint != expected:
        raise ValueError("full-motion run configuration or source provenance differs")
    _require_fingerprint(
        _read_json(run_dir / "training_complete.json"),
        fingerprint,
        run_dir / "training_complete.json",
    )
    marker = run_dir / "evaluation_complete.json"
    if marker.is_file():
        if not resume:
            raise FileExistsError("evaluation is complete; pass --resume to reuse")
        _require_fingerprint(_read_json(marker), fingerprint, marker)
        return _read_json(run_dir / "evaluation_summary.json")

    normalizer = _load_train_normalizer(motion_stats_path)
    data = load_full_motion_data(e5_run_dir, normalizer, splits=("test",))
    if formal and len(data) != 100:
        raise ValueError(f"formal test requires 100 samples, got {len(data)}")
    atomic_write_json(run_dir / "data_audit/test.json", data_audit(data))
    rows = _baseline_rows(data, normalizer)
    transports = [item.transport for item in data]
    device = _resolve_device(settings.device)
    for channel_uses in settings.channel_uses:
        for seed in settings.seeds:
            checkpoint = load_checkpoint(
                run_dir / "models" / f"c_{channel_uses}" / f"seed_{seed}" / "best.pt",
                expected_fingerprint=fingerprint,
                map_location=device,
            )
            model = _build_model(settings, channel_uses, seed, device)
            model.load_state_dict(checkpoint["model_state"])
            channel_rows = _evaluate_one_model(
                model,
                transports,
                settings,
                normalizer.std,
                channel_uses=channel_uses,
                model_seed=seed,
                device=device,
            )
            rows.extend(_rename_channel_row(row) for row in channel_rows)
    _atomic_write_jsonl(run_dir / "test_metrics.jsonl", rows)
    summary = _summarize(rows, settings)
    atomic_write_json(run_dir / "evaluation_summary.json", summary)
    _write_csv(
        run_dir / "evaluation_summary.csv",
        summary["seed_aggregate"],
        (
            "condition",
            "channel_uses",
            "snr_db",
            "model_seed_count",
            "normalized_motion_mse_mean",
            "normalized_motion_mse_std",
            "l1_mean",
            "l1_std",
            "rmse_mean",
            "rmse_std",
            "velocity_l1_mean",
            "velocity_l1_std",
            "improves_audio_prediction_l1",
        ),
    )
    atomic_write_json(
        marker,
        {
            "experiment_fingerprint": fingerprint,
            "status": "complete",
            "test_sample_count": len(data),
            "result_count": len(rows),
        },
    )
    return summary


def _baseline_rows(
    data: Sequence[FullMotionData],
    normalizer: MotionNormalizer,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in data:
        source = item.source
        for condition, candidate in (
            ("audio_prediction", source.prediction),
            ("full_motion_oracle", source.target),
        ):
            metrics = compute_motion_metrics(
                source.target[source.valid_mask],
                candidate[source.valid_mask],
            )
            difference = (candidate - source.target) / normalizer.std
            rows.append(
                {
                    "sample_id": source.sample_id,
                    "speaker_id": source.speaker_id,
                    "split": source.split,
                    "representation": _REPRESENTATION,
                    "condition": condition,
                    "channel_uses": None,
                    "model_seed": None,
                    "snr_db": None,
                    "noise_seed": None,
                    "normalized_motion_mse": float(
                        np.square(
                            difference[source.transmission_mask],
                            dtype=np.float64,
                        ).mean()
                    ),
                    **metrics.to_dict(),
                }
            )
    return rows


def _rename_channel_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(row)
    output["representation"] = _REPRESENTATION
    output["normalized_motion_mse"] = output.pop("normalized_residual_mse")
    return output


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    settings: JSCCSettings,
) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["condition"],
            row.get("channel_uses"),
            row.get("model_seed"),
            row.get("snr_db"),
        )
        grouped[key].append(row)
    groups: list[dict[str, Any]] = []
    for key, members in sorted(grouped.items(), key=lambda item: str(item[0])):
        condition, channel_uses, model_seed, snr_db = key
        groups.append(
            {
                "condition": condition,
                "channel_uses": channel_uses,
                "model_seed": model_seed,
                "snr_db": snr_db,
                "sample_noise_realization_count": len(members),
                **{
                    metric: float(np.mean([float(row[metric]) for row in members]))
                    for metric in _METRICS
                },
            }
        )
    aggregate = _aggregate_seeds(groups, settings)
    return {
        "schema_version": 1,
        "status": "evaluation_complete",
        "representation": _REPRESENTATION,
        "channel_backend": settings.channel_backend,
        "channel_model": "complex_awgn",
        "channel_use_unit": "complex_symbols_per_eligible_frame",
        "bitrate_claimed": False,
        "result_count": len(rows),
        "groups": groups,
        "seed_aggregate": aggregate,
    }


def _aggregate_seeds(
    groups: Sequence[Mapping[str, Any]],
    settings: JSCCSettings,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    audio_l1 = next(float(row["l1"]) for row in groups if row["condition"] == "audio_prediction")
    for condition in ("audio_prediction", "full_motion_oracle"):
        source = next(row for row in groups if row["condition"] == condition)
        output.append(
            {
                "condition": condition,
                "channel_uses": None,
                "snr_db": None,
                "model_seed_count": 0,
                "improves_audio_prediction_l1": float(source["l1"]) < audio_l1,
                **{
                    f"{metric}_{suffix}": (float(source[metric]) if suffix == "mean" else 0.0)
                    for metric in _METRICS
                    for suffix in ("mean", "std")
                },
            }
        )
    for channel_uses in settings.channel_uses:
        for condition, snr_values in (
            ("noiseless_autoencoder", (None,)),
            ("jscc_awgn", settings.test_snr_db),
        ):
            for snr_db in snr_values:
                members = [
                    row
                    for row in groups
                    if row["condition"] == condition
                    and row["channel_uses"] == channel_uses
                    and row["snr_db"] == snr_db
                ]
                if len(members) != len(settings.seeds):
                    raise ValueError("full-motion evaluation is missing a model seed")
                record: dict[str, Any] = {
                    "condition": condition,
                    "channel_uses": channel_uses,
                    "snr_db": snr_db,
                    "model_seed_count": len(members),
                }
                for metric in _METRICS:
                    values = [float(row[metric]) for row in members]
                    record[f"{metric}_mean"] = float(np.mean(values))
                    record[f"{metric}_std"] = float(np.std(values))
                record["improves_audio_prediction_l1"] = record["l1_mean"] < audio_l1
                output.append(record)
    return output


def _input_provenance(
    settings: JSCCSettings,
    motion_stats_path: Path,
    e5_run_dir: Path,
) -> dict[str, Any]:
    metadata_path = e5_run_dir / "run_metadata.json"
    training_path = e5_run_dir / "training_complete.json"
    evaluation_path = e5_run_dir / "evaluation_complete.json"
    metadata = _read_json(metadata_path)
    fingerprint = str(metadata.get("experiment_fingerprint", ""))
    if not fingerprint:
        raise ValueError("source E5 run has no fingerprint")
    for path in (training_path, evaluation_path):
        marker = _read_json(path)
        if (
            marker.get("status") != "complete"
            or marker.get("experiment_fingerprint") != fingerprint
        ):
            raise ValueError(f"source E5 run is incomplete: {path}")
    return {
        "representation": _REPRESENTATION,
        "e5_experiment_fingerprint": fingerprint,
        "e5_metadata_sha256": file_sha256(metadata_path),
        "e5_training_complete_sha256": file_sha256(training_path),
        "e5_evaluation_complete_sha256": file_sha256(evaluation_path),
        "motion_stats_sha256": file_sha256(motion_stats_path),
        "channel_backend": settings.channel_backend,
        "source_reuse": "targets_predictions_and_identity_only",
    }


def _fingerprint(
    settings: JSCCSettings,
    inputs: Mapping[str, Any],
) -> str:
    return config_fingerprint(
        {
            "representation": _REPRESENTATION,
            "config": settings.config,
            "inputs": dict(inputs),
        }
    )


def _prepare_run(
    run_dir: Path,
    settings: JSCCSettings,
    inputs: Mapping[str, Any],
    fingerprint: str,
    *,
    resume: bool,
) -> None:
    metadata_path = run_dir / "run_metadata.json"
    if run_dir.exists():
        if not resume:
            raise FileExistsError(f"run directory already exists: {run_dir}")
        _require_fingerprint(_read_json(metadata_path), fingerprint, metadata_path)
        return
    if resume:
        raise FileNotFoundError(f"cannot resume missing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    atomic_write_json(run_dir / "resolved_config.json", dict(settings.config))
    atomic_write_json(run_dir / "input_provenance.json", dict(inputs))
    atomic_write_json(run_dir / "environment.json", _environment())
    atomic_write_json(
        metadata_path,
        {
            "experiment_fingerprint": fingerprint,
            "representation": _REPRESENTATION,
            "git_commit": _git_commit(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _load_train_normalizer(path: Path) -> MotionNormalizer:
    normalizer = load_motion_normalizer(path)
    if normalizer.scope != "train_stats":
        raise ValueError("full-motion JSCC requires frozen train_stats")
    return normalizer
