"""Export full-motion JSCC candidates for the frozen LivePortrait evaluator."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from av_semcom.data.preprocessing import atomic_write_json, config_fingerprint
from av_semcom.metrics.motion import compute_motion_metrics
from av_semcom.models.full_motion.data import FullMotionData, load_full_motion_data
from av_semcom.models.jscc.candidates import (
    JSCCCandidateBundle,
    JSCCCondition,
    condition_id,
    load_candidate_bundle,
    save_candidate_bundle,
)
from av_semcom.models.jscc.config import JSCCReconstructionSettings, JSCCSettings
from av_semcom.models.jscc.experiment import _derived_noise_seed
from av_semcom.models.jscc.export import (
    _load_model,
    select_validation_model_seeds,
)
from av_semcom.models.motion.perturbations import load_motion_normalizer
from av_semcom.models.predictor.artifacts import file_sha256


def export_full_motion_candidates(
    settings: JSCCSettings,
    reconstruction: JSCCReconstructionSettings,
    motion_stats_path: Path,
    e5_run_dir: Path,
    run_dir: Path,
    *,
    resume: bool = False,
    formal: bool = True,
) -> dict[str, Any]:
    """Export the validation-selected full-motion models once."""

    if formal:
        settings.require_formal_backend()
    run_dir = run_dir.resolve()
    metadata = _read_json(run_dir / "run_metadata.json")
    fingerprint = str(metadata.get("experiment_fingerprint", ""))
    if not fingerprint:
        raise ValueError("full-motion run has no fingerprint")
    for filename in ("training_complete.json", "evaluation_complete.json"):
        marker = _read_json(run_dir / filename)
        if (
            marker.get("status") != "complete"
            or marker.get("experiment_fingerprint") != fingerprint
        ):
            raise ValueError(f"full-motion run is incomplete: {filename}")
    selected = select_validation_model_seeds(
        _read_json(run_dir / "training_summary.json"),
        settings.channel_uses,
    )
    metrics_path = run_dir / "test_metrics.jsonl"
    metric_rows = _read_jsonl(metrics_path)
    metric_index = _metric_index(metric_rows)
    candidate_fingerprint = config_fingerprint(
        {
            "experiment_fingerprint": fingerprint,
            "source_test_metrics_sha256": file_sha256(metrics_path),
            "source_e5_experiment_fingerprint": _read_json(e5_run_dir / "run_metadata.json")[
                "experiment_fingerprint"
            ],
            "split": reconstruction.split,
            "noise_seed": reconstruction.noise_seed,
            "channel_uses": settings.channel_uses,
            "test_snr_db": settings.test_snr_db,
            "selected_model_seeds": selected,
            "representation": "train_standardized_full_18d_motion",
        }
    )
    output_root = run_dir / "reconstruction_candidates"
    complete_path = output_root / "complete.json"
    if complete_path.is_file():
        if not resume:
            raise FileExistsError("full-motion candidates are complete; pass --resume")
        complete = _read_json(complete_path)
        if complete.get("candidate_fingerprint") != candidate_fingerprint:
            raise ValueError("full-motion candidate fingerprint mismatch")
        return complete

    normalizer = load_motion_normalizer(motion_stats_path)
    data = load_full_motion_data(
        e5_run_dir.resolve(),
        normalizer,
        splits=(reconstruction.split,),
    )
    device = _resolve_device(settings.device)
    models = {
        uses: _load_model(
            settings,
            run_dir,
            fingerprint,
            uses,
            selected[uses],
            device,
        )
        for uses in settings.channel_uses
    }
    maximum_difference = 0.0
    for position, item in enumerate(data):
        path = output_root / reconstruction.split / f"{item.source.sample_id}.npz"
        if path.is_file():
            if not resume:
                raise FileExistsError(f"candidate already exists: {path}")
            load_candidate_bundle(path, expected_fingerprint=candidate_fingerprint)
            continue
        bundle = _build_bundle(
            item,
            position,
            models,
            selected,
            settings,
            reconstruction,
            normalizer.std,
            fingerprint,
            candidate_fingerprint,
        )
        maximum_difference = max(
            maximum_difference,
            _cross_check(bundle, item, metric_index),
        )
        save_candidate_bundle(path, bundle)
    if maximum_difference > 1e-9:
        raise ValueError(f"candidate metric cross-check failed: {maximum_difference}")
    runtime = {
        "experiment_fingerprint": fingerprint,
        "candidate_fingerprint": candidate_fingerprint,
        "source_test_metrics_sha256": file_sha256(metrics_path),
        "representation": "train_standardized_full_18d_motion",
        "selection_rule": "minimum_validation_normalized_mse_per_channel_use",
        "selected_model_seeds": {str(key): value for key, value in selected.items()},
        "split": reconstruction.split,
        "noise_seed": reconstruction.noise_seed,
        "sample_count": len(data),
        "condition_count_per_sample": 2
        + len(settings.channel_uses) * (1 + len(settings.test_snr_db)),
        "maximum_motion_metric_difference": maximum_difference,
    }
    atomic_write_json(output_root / "runtime.json", runtime)
    atomic_write_json(complete_path, {**runtime, "status": "complete"})
    return {**runtime, "status": "complete"}


@torch.no_grad()
def _build_bundle(
    item: FullMotionData,
    example_index: int,
    models: Mapping[int, torch.nn.Module],
    selected_seeds: Mapping[int, int],
    settings: JSCCSettings,
    reconstruction: JSCCReconstructionSettings,
    motion_std: np.ndarray,
    experiment_fingerprint: str,
    candidate_fingerprint: str,
) -> JSCCCandidateBundle:
    source = item.source
    transport = item.transport
    conditions = [
        JSCCCondition(
            condition_id="audio_prediction",
            family="audio_prediction",
            channel_uses=None,
            model_seed=None,
            snr_db=None,
            noise_seed=None,
        ),
        JSCCCondition(
            condition_id="full_motion_oracle",
            family="full_motion_oracle",
            channel_uses=None,
            model_seed=None,
            snr_db=None,
            noise_seed=None,
        ),
    ]
    vectors = [source.prediction.copy(), source.target.copy()]
    normalized = torch.from_numpy(transport.normalized_residual).unsqueeze(0)
    mask = torch.from_numpy(transport.transmission_mask).unsqueeze(0)
    for uses in settings.channel_uses:
        model = models[uses]
        device = next(model.parameters()).device
        active = normalized.to(device)
        active_mask = mask.to(device)
        seed = selected_seeds[uses]
        noiseless = model(active, active_mask, 0.0, add_noise=False)
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
        vectors.append(
            _reconstruct(
                transport,
                noiseless.decoded_residual[0].float().cpu().numpy(),
                motion_std,
            )
        )
        for snr_index, snr_db in enumerate(settings.test_snr_db):
            result = model(
                active,
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
                _reconstruct(
                    transport,
                    result.decoded_residual[0].float().cpu().numpy(),
                    motion_std,
                )
            )
    return JSCCCandidateBundle(
        sample_id=source.sample_id,
        split=source.split,
        speaker_id=source.speaker_id,
        experiment_fingerprint=experiment_fingerprint,
        candidate_fingerprint=candidate_fingerprint,
        conditions=tuple(conditions),
        vectors=np.stack(vectors).astype(np.float32),
        valid_mask=source.valid_mask,
    )


def _reconstruct(
    transport: Any,
    decoded_normalized_motion: np.ndarray,
    motion_std: np.ndarray,
) -> np.ndarray:
    decoded = np.asarray(decoded_normalized_motion, dtype=np.float32).copy()
    decoded[~transport.transmission_mask] = 0
    vector = transport.prediction + decoded * np.asarray(
        motion_std,
        dtype=np.float32,
    )
    vector[0] = 0
    return vector.astype(np.float32)


def _metric_index(
    rows: list[dict[str, Any]],
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    output: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            row["sample_id"],
            row["condition"],
            row.get("channel_uses"),
            row.get("model_seed"),
            row.get("snr_db"),
            row.get("noise_seed"),
        )
        if key in output:
            raise ValueError(f"duplicate full-motion metric identity: {key}")
        output[key] = row
    return output


def _cross_check(
    bundle: JSCCCandidateBundle,
    item: FullMotionData,
    metric_index: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> float:
    maximum = 0.0
    source = item.source
    for condition, vector in zip(bundle.conditions, bundle.vectors, strict=True):
        values = compute_motion_metrics(
            source.target[source.valid_mask],
            vector[source.valid_mask],
        ).to_dict()
        key = (
            source.sample_id,
            condition.family,
            condition.channel_uses,
            condition.model_seed,
            condition.snr_db,
            condition.noise_seed,
        )
        row = metric_index.get(key)
        if row is None:
            raise ValueError(f"no metric for candidate {condition.condition_id}")
        for metric, value in values.items():
            maximum = max(maximum, abs(value - float(row[metric])))
    return maximum


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA candidate export was requested but is unavailable")
    return device


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
