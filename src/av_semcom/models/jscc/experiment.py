"""Training and evaluation for the Sionna-based E5 residual JSCC baseline."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from av_semcom.data.grid import GridSample
from av_semcom.data.preprocessing import atomic_write_json, config_fingerprint
from av_semcom.metrics.motion import compute_motion_metrics
from av_semcom.models.jscc.config import JSCCSettings
from av_semcom.models.jscc.data import (
    ResidualDataset,
    ResidualExample,
    load_residual_example,
    prepare_residual_examples,
    save_residual_example,
    select_best_e3_seed,
)
from av_semcom.models.jscc.model import ResidualJSCC, masked_residual_mse
from av_semcom.models.motion.perturbations import load_motion_normalizer
from av_semcom.models.predictor.artifacts import (
    atomic_save_checkpoint,
    file_sha256,
    load_checkpoint,
)
from av_semcom.models.predictor.config import AudioMotionSettings
from av_semcom.utils.reproducibility import seed_everything


def run_jscc_training(
    settings: JSCCSettings,
    predictor_settings: AudioMotionSettings,
    samples: Sequence[GridSample],
    e3_run_dir: Path,
    *,
    run_directory: Path | None = None,
    resume: bool = False,
    formal: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Train every configured channel-use and model-seed condition."""

    if formal:
        settings.require_formal_backend()
    inputs = _input_provenance(settings, predictor_settings, e3_run_dir)
    fingerprint = config_fingerprint({"config": settings.config, "inputs": inputs})
    run_dir = run_directory or _new_run_directory(settings.output_root)
    _prepare_run_directory(
        run_dir,
        settings,
        inputs,
        fingerprint,
        resume=resume,
    )
    examples = _load_or_prepare_examples(
        run_dir,
        samples,
        predictor_settings,
        e3_run_dir,
        splits=("train", "validation"),
        fingerprint=fingerprint,
        resume=resume,
    )
    train_examples = [example for example in examples if example.split == "train"]
    validation_examples = [example for example in examples if example.split == "validation"]
    if not train_examples or not validation_examples:
        raise ValueError("E5 training requires non-empty train and validation splits")
    if set(example.speaker_id for example in train_examples) & set(
        example.speaker_id for example in validation_examples
    ):
        raise ValueError("speaker leakage between train and validation residuals")

    model_rows: list[dict[str, Any]] = []
    for channel_uses in settings.channel_uses:
        for seed in settings.seeds:
            model_dir = run_dir / "models" / f"c_{channel_uses}" / f"seed_{seed}"
            marker = model_dir / "complete.json"
            if resume and marker.is_file():
                payload = _read_json(marker)
                _require_fingerprint(payload, fingerprint, marker)
                model_rows.append(payload)
                continue
            model_rows.append(
                _train_one_model(
                    settings,
                    train_examples,
                    validation_examples,
                    channel_uses,
                    seed,
                    model_dir,
                    fingerprint,
                )
            )
    summary = {
        "schema_version": 1,
        "status": "training_complete",
        "experiment_fingerprint": fingerprint,
        "channel_backend": settings.channel_backend,
        "channel_model": "complex_awgn",
        "channel_use_unit": "complex_symbols_per_eligible_frame",
        "semantic_compression_ratio_definition": "complex_channel_uses / 18",
        "models": model_rows,
    }
    atomic_write_json(run_dir / "training_summary.json", summary)
    _write_training_summary_csv(run_dir / "training_summary.csv", model_rows)
    atomic_write_json(
        run_dir / "training_complete.json",
        {
            "experiment_fingerprint": fingerprint,
            "model_count": len(model_rows),
            "status": "complete",
        },
    )
    return run_dir, summary


def run_jscc_evaluation(
    settings: JSCCSettings,
    predictor_settings: AudioMotionSettings,
    samples: Sequence[GridSample],
    e3_run_dir: Path,
    run_dir: Path,
    *,
    resume: bool = False,
    formal: bool = True,
) -> dict[str, Any]:
    """Evaluate frozen checkpoints on test only, without model selection."""

    if formal:
        settings.require_formal_backend()
    run_dir = run_dir.resolve()
    metadata = _read_json(run_dir / "run_metadata.json")
    fingerprint = str(metadata.get("experiment_fingerprint", ""))
    expected_inputs = _input_provenance(settings, predictor_settings, e3_run_dir)
    expected = config_fingerprint({"config": settings.config, "inputs": expected_inputs})
    if fingerprint != expected:
        raise ValueError("training run configuration or input provenance does not match")
    complete = _read_json(run_dir / "training_complete.json")
    _require_fingerprint(complete, fingerprint, run_dir / "training_complete.json")
    marker = run_dir / "evaluation_complete.json"
    if marker.is_file() and not resume:
        raise FileExistsError(
            "evaluation is already complete; pass --resume to verify and reuse it"
        )
    if resume and marker.is_file():
        _require_fingerprint(_read_json(marker), fingerprint, marker)
        return _read_json(run_dir / "evaluation_summary.json")

    test_examples = _load_or_prepare_examples(
        run_dir,
        samples,
        predictor_settings,
        e3_run_dir,
        splits=("test",),
        fingerprint=fingerprint,
        resume=resume,
    )
    rows = _evaluate_baselines(test_examples)
    motion_std = load_motion_normalizer(predictor_settings.motion_stats_path).std
    device = _resolve_device(settings.device)
    for channel_uses in settings.channel_uses:
        for seed in settings.seeds:
            model_dir = run_dir / "models" / f"c_{channel_uses}" / f"seed_{seed}"
            checkpoint = load_checkpoint(
                model_dir / "best.pt",
                expected_fingerprint=fingerprint,
                map_location=device,
            )
            model = _build_model(settings, channel_uses, seed, device)
            model.load_state_dict(checkpoint["model_state"])
            rows.extend(
                _evaluate_one_model(
                    model,
                    test_examples,
                    settings,
                    motion_std,
                    channel_uses=channel_uses,
                    model_seed=seed,
                    device=device,
                )
            )
    _atomic_write_jsonl(run_dir / "test_metrics.jsonl", rows)
    summary = _summarize_test_rows(rows, settings)
    atomic_write_json(run_dir / "evaluation_summary.json", summary)
    _write_evaluation_summary_csv(
        run_dir / "evaluation_summary.csv",
        summary["seed_aggregate"],
    )
    atomic_write_json(
        marker,
        {
            "experiment_fingerprint": fingerprint,
            "status": "complete",
            "test_sample_count": len(test_examples),
            "result_count": len(rows),
        },
    )
    return summary


def write_jscc_report(settings: JSCCSettings, run_dir: Path) -> dict[str, Any]:
    """Derive multi-seed tables and curves from an immutable test JSONL."""

    run_dir = run_dir.resolve()
    marker_path = run_dir / "evaluation_complete.json"
    marker = _read_json(marker_path)
    metrics_path = run_dir / "test_metrics.jsonl"
    rows = _read_jsonl(metrics_path)
    if len(rows) != int(marker.get("result_count", -1)):
        raise ValueError("test metric row count does not match completion marker")
    input_provenance = _read_json(run_dir / "input_provenance.json")
    if input_provenance.get("channel_backend") != settings.channel_backend:
        raise ValueError("report channel backend does not match training run")
    summary = _summarize_test_rows(rows, settings)
    summary.update(
        {
            "experiment_fingerprint": marker["experiment_fingerprint"],
            "source_metrics_sha256": file_sha256(metrics_path),
            "report_git_commit": _git_commit(),
            "report_created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    atomic_write_json(run_dir / "report_summary.json", summary)
    _write_evaluation_summary_csv(
        run_dir / "report_summary.csv",
        summary["seed_aggregate"],
    )
    _write_snr_plots(run_dir / "plots", summary)
    return summary


def _train_one_model(
    settings: JSCCSettings,
    train_examples: Sequence[ResidualExample],
    validation_examples: Sequence[ResidualExample],
    channel_uses: int,
    seed: int,
    model_dir: Path,
    fingerprint: str,
) -> dict[str, Any]:
    seed_everything(seed, deterministic=settings.deterministic)
    device = _resolve_device(settings.device)
    model = _build_model(settings, channel_uses, seed, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        ResidualDataset(train_examples),
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        ResidualDataset(validation_examples),
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=device.type == "cuda",
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "best.pt"
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    snr_generator = torch.Generator().manual_seed(seed + 10_000)
    for epoch in range(1, settings.max_epochs + 1):
        train_loss = _train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            settings,
            snr_generator,
        )
        validation_loss = _validation_loss(
            model,
            validation_loader,
            device,
            settings,
            model_seed=seed,
        )
        if not np.isfinite(train_loss) or not np.isfinite(validation_loss):
            raise RuntimeError(
                f"non-finite JSCC loss at C={channel_uses}, seed={seed}, epoch={epoch}"
            )
        row = {
            "epoch": epoch,
            "train_normalized_mse": train_loss,
            "validation_normalized_mse": validation_loss,
        }
        history.append(row)
        print(
            f"[residual-jscc] C={channel_uses} seed={seed} epoch={epoch} "
            f"train_mse={train_loss:.6f} validation_mse={validation_loss:.6f}",
            flush=True,
        )
        if validation_loss < best_loss - settings.early_stopping_min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            atomic_save_checkpoint(
                checkpoint_path,
                {
                    "experiment_fingerprint": fingerprint,
                    "channel_backend": settings.channel_backend,
                    "channel_uses": channel_uses,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_normalized_mse": validation_loss,
                    "model_config": _model_config(settings, channel_uses),
                    "model_state": model.state_dict(),
                },
            )
        else:
            stale_epochs += 1
            if stale_epochs >= settings.early_stopping_patience:
                break
    _atomic_write_jsonl(model_dir / "history.jsonl", history)
    result = {
        "experiment_fingerprint": fingerprint,
        "channel_backend": settings.channel_backend,
        "channel_uses": channel_uses,
        "real_degrees_of_freedom": 2 * channel_uses,
        "semantic_compression_ratio": channel_uses / 18.0,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_normalized_mse": best_loss,
        "epoch_count": len(history),
    }
    atomic_write_json(model_dir / "complete.json", result)
    return result


def _train_epoch(
    model: ResidualJSCC,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    settings: JSCCSettings,
    snr_generator: torch.Generator,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        residual = batch["residual"].to(device)
        mask = batch["mask"].to(device)
        snr = (
            torch.rand(
                (residual.shape[0], 1, 1),
                generator=snr_generator,
                dtype=torch.float32,
            )
            * (settings.train_snr_max_db - settings.train_snr_min_db)
            + settings.train_snr_min_db
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        result = model(residual, mask, snr)
        loss = masked_residual_mse(result.decoded_residual, residual, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
        optimizer.step()
        batch_size = residual.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
    return total_loss / total_samples


@torch.no_grad()
def _validation_loss(
    model: ResidualJSCC,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    settings: JSCCSettings,
    *,
    model_seed: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for snr_index, snr_db in enumerate(settings.validation_snr_db):
        for noise_seed in settings.noise_seeds:
            for batch_index, batch in enumerate(loader):
                residual = batch["residual"].to(device)
                mask = batch["mask"].to(device)
                result = model(
                    residual,
                    mask,
                    snr_db,
                    noise_seed=_derived_noise_seed(
                        model_seed,
                        noise_seed,
                        snr_index,
                        batch_index,
                    ),
                )
                losses.append(
                    float(masked_residual_mse(result.decoded_residual, residual, mask).cpu())
                )
    return float(np.mean(losses))


@torch.no_grad()
def _evaluate_one_model(
    model: ResidualJSCC,
    examples: Sequence[ResidualExample],
    settings: JSCCSettings,
    motion_std: np.ndarray,
    *,
    channel_uses: int,
    model_seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    std = np.asarray(motion_std, dtype=np.float32)
    for example_index, example in enumerate(examples):
        residual = torch.from_numpy(example.normalized_residual).unsqueeze(0).to(device)
        mask = torch.from_numpy(example.transmission_mask).unsqueeze(0).to(device)
        noiseless = model(residual, mask, 0.0, add_noise=False)
        decoded = noiseless.decoded_residual[0].cpu().numpy().astype(np.float32)
        rows.append(
            _metric_row(
                example,
                decoded,
                std,
                condition="noiseless_autoencoder",
                channel_uses=channel_uses,
                model_seed=model_seed,
                snr_db=None,
                noise_seed=None,
            )
        )
        for snr_index, snr_db in enumerate(settings.test_snr_db):
            for noise_seed in settings.noise_seeds:
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


def _evaluate_baselines(examples: Sequence[ResidualExample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        for condition, candidate in (
            ("prediction_only", example.prediction),
            ("full_residual_oracle", example.target),
        ):
            metrics = compute_motion_metrics(
                example.target[example.valid_mask],
                candidate[example.valid_mask],
            )
            residual_error = (
                example.normalized_residual
                if condition == "prediction_only"
                else np.zeros_like(example.normalized_residual)
            )
            rows.append(
                {
                    "sample_id": example.sample_id,
                    "speaker_id": example.speaker_id,
                    "split": example.split,
                    "condition": condition,
                    "channel_uses": None,
                    "model_seed": None,
                    "snr_db": None,
                    "noise_seed": None,
                    "normalized_residual_mse": float(
                        np.square(
                            residual_error[example.transmission_mask],
                            dtype=np.float64,
                        ).mean()
                    ),
                    **metrics.to_dict(),
                }
            )
    return rows


def _metric_row(
    example: ResidualExample,
    decoded_normalized_residual: np.ndarray,
    motion_std: np.ndarray,
    *,
    condition: str,
    channel_uses: int,
    model_seed: int,
    snr_db: float | None,
    noise_seed: int | None,
) -> dict[str, Any]:
    decoded = np.asarray(decoded_normalized_residual, dtype=np.float32).copy()
    decoded[~example.transmission_mask] = 0
    reconstructed = example.prediction + decoded * motion_std
    reconstructed[0] = 0
    metrics = compute_motion_metrics(
        example.target[example.valid_mask],
        reconstructed[example.valid_mask],
    )
    difference = decoded - example.normalized_residual
    return {
        "sample_id": example.sample_id,
        "speaker_id": example.speaker_id,
        "split": example.split,
        "condition": condition,
        "channel_uses": channel_uses,
        "real_degrees_of_freedom": 2 * channel_uses,
        "semantic_compression_ratio": channel_uses / 18.0,
        "model_seed": model_seed,
        "snr_db": snr_db,
        "noise_seed": noise_seed,
        "normalized_residual_mse": float(
            np.square(difference[example.transmission_mask], dtype=np.float64).mean()
        ),
        **metrics.to_dict(),
    }


def _summarize_test_rows(
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
                    for metric in ("normalized_residual_mse", "l1", "rmse", "velocity_l1")
                },
            }
        )
    seed_aggregate = _aggregate_model_seeds(groups, settings)
    return {
        "schema_version": 2,
        "status": "evaluation_complete",
        "channel_backend": settings.channel_backend,
        "channel_model": "complex_awgn",
        "channel_use_unit": "complex_symbols_per_eligible_frame",
        "semantic_compression_ratio_definition": "complex_channel_uses / 18",
        "bitrate_claimed": False,
        "result_count": len(rows),
        "groups": groups,
        "seed_aggregate": seed_aggregate,
    }


def _aggregate_model_seeds(
    groups: Sequence[Mapping[str, Any]],
    settings: JSCCSettings,
) -> list[dict[str, Any]]:
    metrics = ("normalized_residual_mse", "l1", "rmse", "velocity_l1")
    aggregate: list[dict[str, Any]] = []
    prediction_l1 = next(
        float(group["l1"]) for group in groups if group["condition"] == "prediction_only"
    )
    for condition in ("prediction_only", "full_residual_oracle"):
        group = next(item for item in groups if item["condition"] == condition)
        aggregate.append(
            {
                "condition": condition,
                "channel_uses": None,
                "snr_db": None,
                "model_seed_count": 0,
                "improves_prediction_only_l1": float(group["l1"]) < prediction_l1,
                **{
                    f"{metric}_{suffix}": float(group[metric]) if suffix == "mean" else 0.0
                    for metric in metrics
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
                    group
                    for group in groups
                    if group["condition"] == condition
                    and group["channel_uses"] == channel_uses
                    and group["snr_db"] == snr_db
                ]
                if len(members) != len(settings.seeds):
                    raise ValueError(
                        f"expected {len(settings.seeds)} model seeds for "
                        f"{condition}, C={channel_uses}, SNR={snr_db}"
                    )
                row: dict[str, Any] = {
                    "condition": condition,
                    "channel_uses": channel_uses,
                    "real_degrees_of_freedom": 2 * channel_uses,
                    "semantic_compression_ratio": channel_uses / 18.0,
                    "snr_db": snr_db,
                    "model_seed_count": len(members),
                }
                for metric in metrics:
                    values = [float(member[metric]) for member in members]
                    row[f"{metric}_mean"] = float(np.mean(values))
                    row[f"{metric}_std"] = float(np.std(values))
                row["improves_prediction_only_l1"] = row["l1_mean"] < prediction_l1
                aggregate.append(row)
    return aggregate


def _load_or_prepare_examples(
    run_dir: Path,
    samples: Sequence[GridSample],
    predictor_settings: AudioMotionSettings,
    e3_run_dir: Path,
    *,
    splits: Sequence[str],
    fingerprint: str,
    resume: bool,
) -> list[ResidualExample]:
    scope = "_".join(splits)
    marker = run_dir / "residual_data" / f"{scope}_complete.json"
    if resume and marker.is_file():
        payload = _read_json(marker)
        _require_fingerprint(payload, fingerprint, marker)
        paths = sorted((run_dir / "residual_data").glob("*/*.npz"))
        wanted = set(splits)
        examples = [
            load_residual_example(path, expected_fingerprint=fingerprint)
            for path in paths
            if path.parent.name in wanted
        ]
        if len(examples) != int(payload["sample_count"]):
            raise ValueError(f"incomplete residual cache for {scope}")
        return examples
    examples, audit = prepare_residual_examples(
        samples,
        predictor_settings,
        e3_run_dir,
        splits=splits,
    )
    for example in examples:
        save_residual_example(
            run_dir / "residual_data" / example.split / f"{example.sample_id}.npz",
            example,
            experiment_fingerprint=fingerprint,
        )
    atomic_write_json(
        run_dir / "data_audit" / f"{scope}.json",
        audit.to_dict(),
    )
    atomic_write_json(
        marker,
        {
            "experiment_fingerprint": fingerprint,
            "splits": list(splits),
            "sample_count": len(examples),
        },
    )
    return examples


def _input_provenance(
    settings: JSCCSettings,
    predictor_settings: AudioMotionSettings,
    e3_run_dir: Path,
) -> dict[str, Any]:
    e3_run_dir = e3_run_dir.resolve()
    experiment = _read_json(e3_run_dir / "experiment.json")
    if experiment.get("status") != "complete":
        raise ValueError("E3 experiment must be complete")
    selected_seed = select_best_e3_seed(e3_run_dir)
    return {
        "manifest_sha256": file_sha256(predictor_settings.data_settings.manifest_path),
        "motion_stats_sha256": file_sha256(predictor_settings.motion_stats_path),
        "e3_experiment_fingerprint": experiment["experiment_fingerprint"],
        "e3_audio_stats_sha256": file_sha256(e3_run_dir / "audio_stats.json"),
        "e3_selected_seed": selected_seed,
        "e3_seed_selection": "minimum_validation_l1_only",
        "e3_checkpoint_sha256": file_sha256(e3_run_dir / f"seed_{selected_seed}" / "best.pt"),
        "sionna_required_version": "2.0.1",
        "channel_backend": settings.channel_backend,
    }


def _prepare_run_directory(
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
        metadata = _read_json(metadata_path)
        _require_fingerprint(metadata, fingerprint, metadata_path)
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
            "git_commit": _git_commit(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _build_model(
    settings: JSCCSettings,
    channel_uses: int,
    seed: int,
    device: torch.device,
) -> ResidualJSCC:
    from av_semcom.channel.awgn import build_awgn_channel

    channel = build_awgn_channel(
        settings.channel_backend,
        device=str(device),
        seed=seed,
    )
    return ResidualJSCC(
        channel=channel,
        **_model_config(settings, channel_uses),
    ).to(device)


def _model_config(settings: JSCCSettings, channel_uses: int) -> dict[str, Any]:
    return {
        "input_dim": settings.input_dim,
        "hidden_dim": settings.hidden_dim,
        "channel_uses": channel_uses,
        "target_power": settings.target_power,
    }


def _derived_noise_seed(
    model_seed: int,
    noise_seed: int,
    condition_index: int,
    item_index: int,
) -> int:
    return (model_seed * 1_000_003 + noise_seed * 10_007 + condition_index * 101 + item_index) % (
        2**31 - 1
    )


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _require_fingerprint(
    payload: Mapping[str, Any],
    fingerprint: str,
    path: Path,
) -> None:
    if payload.get("experiment_fingerprint") != fingerprint:
        raise ValueError(f"artifact fingerprint mismatch: {path}")


def _new_run_directory(root: Path) -> Path:
    return root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    return payload


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_training_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_csv(
        path,
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


def _write_evaluation_summary_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    _write_csv(
        path,
        rows,
        (
            "condition",
            "channel_uses",
            "real_degrees_of_freedom",
            "semantic_compression_ratio",
            "snr_db",
            "model_seed_count",
            "normalized_residual_mse_mean",
            "normalized_residual_mse_std",
            "l1_mean",
            "l1_std",
            "rmse_mean",
            "rmse_std",
            "velocity_l1_mean",
            "velocity_l1_std",
            "improves_prediction_only_l1",
        ),
    )


def _write_snr_plots(path: Path, summary: Mapping[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    aggregate = summary["seed_aggregate"]
    prediction = next(row for row in aggregate if row["condition"] == "prediction_only")
    path.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, filename in (
        ("l1", "Raw motion L1", "motion_l1_vs_snr.png"),
        ("velocity_l1", "Raw motion velocity L1", "velocity_l1_vs_snr.png"),
        (
            "normalized_residual_mse",
            "Normalized residual MSE",
            "residual_mse_vs_snr.png",
        ),
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.axhline(
            float(prediction[f"{metric}_mean"]),
            color="black",
            linestyle="--",
            label="prediction only",
        )
        for channel_uses in sorted(
            {int(row["channel_uses"]) for row in aggregate if row["condition"] == "jscc_awgn"}
        ):
            rows = sorted(
                (
                    row
                    for row in aggregate
                    if row["condition"] == "jscc_awgn" and row["channel_uses"] == channel_uses
                ),
                key=lambda row: float(row["snr_db"]),
            )
            axis.errorbar(
                [row["snr_db"] for row in rows],
                [row[f"{metric}_mean"] for row in rows],
                yerr=[row[f"{metric}_std"] for row in rows],
                marker="o",
                capsize=3,
                label=f"C={channel_uses}",
            )
        axis.set_xlabel("SNR (dB)")
        axis.set_ylabel(ylabel)
        axis.set_title(f"Sionna complex-AWGN residual JSCC: {ylabel}")
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(path / filename, dpi=160)
        plt.close(figure)


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _environment() -> dict[str, Any]:
    try:
        sionna_version = importlib.metadata.version("sionna-no-rt")
    except importlib.metadata.PackageNotFoundError:
        sionna_version = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "sionna_no_rt": sionna_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "device_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
    }
