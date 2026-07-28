"""Export frozen Sionna JSCC motion candidates for the LivePortrait environment."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from av_semcom.data.preprocessing import atomic_write_json, config_fingerprint
from av_semcom.metrics.motion import compute_motion_metrics
from av_semcom.models.jscc.candidates import (
    JSCCCandidateBundle,
    JSCCCondition,
    condition_id,
    load_candidate_bundle,
    save_candidate_bundle,
)
from av_semcom.models.jscc.config import JSCCReconstructionSettings, JSCCSettings
from av_semcom.models.jscc.data import ResidualExample, load_residual_example
from av_semcom.models.jscc.experiment import _build_model, _derived_noise_seed
from av_semcom.models.motion.perturbations import load_motion_normalizer
from av_semcom.models.predictor.artifacts import file_sha256, load_checkpoint
from av_semcom.models.predictor.config import AudioMotionSettings


def select_validation_model_seeds(
    training_summary: Mapping[str, Any],
    channel_uses: Sequence[int],
) -> dict[int, int]:
    """Select one model seed per channel budget using validation MSE only."""

    models = training_summary.get("models")
    if not isinstance(models, list):
        raise ValueError("training summary has no model list")
    selected: dict[int, int] = {}
    for uses in channel_uses:
        candidates = [row for row in models if int(row["channel_uses"]) == uses]
        if not candidates:
            raise ValueError(f"training summary has no model for C={uses}")
        best = min(
            candidates,
            key=lambda row: float(row["best_validation_normalized_mse"]),
        )
        selected[uses] = int(best["seed"])
    return selected


def export_jscc_reconstruction_candidates(
    settings: JSCCSettings,
    reconstruction: JSCCReconstructionSettings,
    predictor_settings: AudioMotionSettings,
    run_dir: Path,
    *,
    resume: bool = False,
    formal: bool = True,
) -> dict[str, Any]:
    """Export all frozen video-evaluation conditions without rendering."""

    if formal:
        settings.require_formal_backend()
    run_dir = run_dir.resolve()
    metadata = _read_json(run_dir / "run_metadata.json")
    experiment_fingerprint = str(metadata.get("experiment_fingerprint", ""))
    if not experiment_fingerprint:
        raise ValueError("JSCC run has no experiment fingerprint")
    training_complete = _read_json(run_dir / "training_complete.json")
    evaluation_complete = _read_json(run_dir / "evaluation_complete.json")
    for path, marker in (
        (run_dir / "training_complete.json", training_complete),
        (run_dir / "evaluation_complete.json", evaluation_complete),
    ):
        if marker.get("experiment_fingerprint") != experiment_fingerprint:
            raise ValueError(f"completion marker fingerprint mismatch: {path}")
    training_summary = _read_json(run_dir / "training_summary.json")
    selected_seeds = select_validation_model_seeds(
        training_summary,
        settings.channel_uses,
    )
    metrics_path = run_dir / "test_metrics.jsonl"
    metric_rows = _read_jsonl(metrics_path)
    candidate_fingerprint = config_fingerprint(
        {
            "experiment_fingerprint": experiment_fingerprint,
            "source_test_metrics_sha256": file_sha256(metrics_path),
            "split": reconstruction.split,
            "noise_seed": reconstruction.noise_seed,
            "channel_uses": settings.channel_uses,
            "test_snr_db": settings.test_snr_db,
            "selected_model_seeds": selected_seeds,
            "selection_rule": "minimum_validation_normalized_mse_per_channel_use",
        }
    )
    output_root = run_dir / "reconstruction_candidates"
    complete_path = output_root / "complete.json"
    if complete_path.is_file() and not resume:
        raise FileExistsError("reconstruction candidates already complete; pass --resume")
    if resume and complete_path.is_file():
        complete = _read_json(complete_path)
        if complete.get("candidate_fingerprint") != candidate_fingerprint:
            raise ValueError("candidate completion fingerprint mismatch")
        return complete

    examples = _load_test_examples(
        run_dir,
        experiment_fingerprint,
        reconstruction.split,
    )
    device = _resolve_device(settings.device)
    models = {
        uses: _load_model(
            settings,
            run_dir,
            experiment_fingerprint,
            uses,
            selected_seeds[uses],
            device,
        )
        for uses in settings.channel_uses
    }
    normalizer = load_motion_normalizer(predictor_settings.motion_stats_path)
    if normalizer.scope != "train_stats":
        raise ValueError("JSCC reconstruction export requires train_stats")
    metric_index = _index_metric_rows(metric_rows)
    maximum_metric_difference = 0.0
    for position, example in enumerate(examples):
        print(
            f"[jscc-export] sample {position + 1}/{len(examples)}: {example.sample_id}",
            flush=True,
        )
        output = output_root / reconstruction.split / f"{example.sample_id}.npz"
        if output.is_file():
            if not resume:
                raise FileExistsError(f"candidate artifact already exists: {output}")
            load_candidate_bundle(output, expected_fingerprint=candidate_fingerprint)
            continue
        bundle = _build_bundle(
            example,
            position,
            models,
            selected_seeds,
            settings,
            reconstruction,
            normalizer.std,
            experiment_fingerprint,
            candidate_fingerprint,
        )
        maximum_metric_difference = max(
            maximum_metric_difference,
            _cross_check_metrics(bundle, example, metric_index),
        )
        save_candidate_bundle(output, bundle)

    runtime = {
        "experiment_fingerprint": experiment_fingerprint,
        "candidate_fingerprint": candidate_fingerprint,
        "source_test_metrics_sha256": file_sha256(metrics_path),
        "selection_rule": "minimum_validation_normalized_mse_per_channel_use",
        "selected_model_seeds": {str(key): value for key, value in selected_seeds.items()},
        "split": reconstruction.split,
        "noise_seed": reconstruction.noise_seed,
        "sample_count": len(examples),
        "condition_count_per_sample": _condition_count(settings),
        "maximum_motion_metric_difference": maximum_metric_difference,
    }
    atomic_write_json(output_root / "runtime.json", runtime)
    atomic_write_json(complete_path, {**runtime, "status": "complete"})
    return {**runtime, "status": "complete"}


def _load_model(
    settings: JSCCSettings,
    run_dir: Path,
    fingerprint: str,
    channel_uses: int,
    seed: int,
    device: torch.device,
) -> torch.nn.Module:
    model = _build_model(settings, channel_uses, seed, device)
    checkpoint = load_checkpoint(
        run_dir / "models" / f"c_{channel_uses}" / f"seed_{seed}" / "best.pt",
        expected_fingerprint=fingerprint,
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


@torch.no_grad()
def _build_bundle(
    example: ResidualExample,
    example_index: int,
    models: Mapping[int, torch.nn.Module],
    selected_seeds: Mapping[int, int],
    settings: JSCCSettings,
    reconstruction: JSCCReconstructionSettings,
    motion_std: np.ndarray,
    experiment_fingerprint: str,
    candidate_fingerprint: str,
) -> JSCCCandidateBundle:
    conditions = [
        JSCCCondition(
            condition_id="prediction_only",
            family="prediction_only",
            channel_uses=None,
            model_seed=None,
            snr_db=None,
            noise_seed=None,
        ),
        JSCCCondition(
            condition_id="full_residual_oracle",
            family="full_residual_oracle",
            channel_uses=None,
            model_seed=None,
            snr_db=None,
            noise_seed=None,
        ),
    ]
    vectors = [example.prediction.copy(), example.target.copy()]
    residual = torch.from_numpy(example.normalized_residual).unsqueeze(0)
    mask = torch.from_numpy(example.transmission_mask).unsqueeze(0)
    std = np.asarray(motion_std, dtype=np.float32)
    for uses in settings.channel_uses:
        model = models[uses]
        device = next(model.parameters()).device
        active_residual = residual.to(device)
        active_mask = mask.to(device)
        seed = selected_seeds[uses]
        noiseless = model(active_residual, active_mask, 0.0, add_noise=False)
        noiseless_vector = _reconstruct_vector(
            example,
            noiseless.decoded_residual[0].float().cpu().numpy(),
            std,
        )
        conditions.append(
            JSCCCondition(
                condition_id=condition_id(
                    "noiseless_autoencoder",
                    channel_uses=uses,
                    model_seed=seed,
                ),
                family="noiseless_autoencoder",
                channel_uses=uses,
                model_seed=seed,
                snr_db=None,
                noise_seed=None,
            )
        )
        vectors.append(noiseless_vector)
        for snr_index, snr_db in enumerate(settings.test_snr_db):
            result = model(
                active_residual,
                active_mask,
                snr_db,
                noise_seed=_derived_noise_seed(
                    seed,
                    reconstruction.noise_seed,
                    snr_index,
                    example_index,
                ),
            )
            conditions.append(
                JSCCCondition(
                    condition_id=condition_id(
                        "jscc_awgn",
                        channel_uses=uses,
                        model_seed=seed,
                        snr_db=snr_db,
                        noise_seed=reconstruction.noise_seed,
                    ),
                    family="jscc_awgn",
                    channel_uses=uses,
                    model_seed=seed,
                    snr_db=snr_db,
                    noise_seed=reconstruction.noise_seed,
                )
            )
            vectors.append(
                _reconstruct_vector(
                    example,
                    result.decoded_residual[0].float().cpu().numpy(),
                    std,
                )
            )
    return JSCCCandidateBundle(
        sample_id=example.sample_id,
        split=example.split,
        speaker_id=example.speaker_id,
        experiment_fingerprint=experiment_fingerprint,
        candidate_fingerprint=candidate_fingerprint,
        conditions=tuple(conditions),
        vectors=np.stack(vectors).astype(np.float32),
        valid_mask=example.valid_mask,
    )


def _reconstruct_vector(
    example: ResidualExample,
    decoded_normalized_residual: np.ndarray,
    motion_std: np.ndarray,
) -> np.ndarray:
    decoded = np.asarray(decoded_normalized_residual, dtype=np.float32).copy()
    decoded[~example.transmission_mask] = 0
    candidate = example.prediction + decoded * motion_std
    candidate[0] = 0
    return candidate.astype(np.float32)


def _load_test_examples(
    run_dir: Path,
    fingerprint: str,
    split: str,
) -> list[ResidualExample]:
    paths = sorted((run_dir / "residual_data" / split).glob("*.npz"))
    if not paths:
        raise ValueError(f"JSCC run contains no cached {split} residuals")
    examples = [load_residual_example(path, expected_fingerprint=fingerprint) for path in paths]
    if any(example.split != split for example in examples):
        raise ValueError("residual cache contains an unexpected split")
    return examples


def _index_metric_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    index: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            row["sample_id"],
            row["condition"],
            row.get("channel_uses"),
            row.get("model_seed"),
            row.get("snr_db"),
            row.get("noise_seed"),
        )
        if key in index:
            raise ValueError(f"duplicate test metric identity: {key}")
        index[key] = row
    return index


def _cross_check_metrics(
    bundle: JSCCCandidateBundle,
    example: ResidualExample,
    metric_index: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> float:
    maximum = 0.0
    for condition, vector in zip(bundle.conditions, bundle.vectors, strict=True):
        metrics = compute_motion_metrics(
            example.target[example.valid_mask],
            vector[example.valid_mask],
        ).to_dict()
        key = (
            example.sample_id,
            condition.family,
            condition.channel_uses,
            condition.model_seed,
            condition.snr_db,
            condition.noise_seed,
        )
        row = metric_index.get(key)
        if row is None:
            raise ValueError(f"no matching frozen test metric for {condition.condition_id}")
        for metric, value in metrics.items():
            maximum = max(maximum, abs(value - float(row[metric])))
    if maximum > 1e-9:
        raise ValueError(f"exported candidates do not reproduce test metrics: {maximum}")
    return maximum


def _condition_count(settings: JSCCSettings) -> int:
    return 2 + len(settings.channel_uses) * (1 + len(settings.test_snr_db))


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export was requested but is unavailable")
    return device


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
