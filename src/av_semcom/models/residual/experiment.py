"""Motion-space prediction-residual analysis for E4."""

from __future__ import annotations

import csv
import hashlib
import json
import math
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

from av_semcom.data.grid import GridSample
from av_semcom.data.preprocessing import atomic_write_json, config_fingerprint
from av_semcom.models.motion.perturbations import load_motion_normalizer
from av_semcom.models.predictor.artifacts import file_sha256, load_prediction
from av_semcom.models.predictor.config import AudioMotionSettings
from av_semcom.models.residual.analysis import (
    ResidualSelection,
    ResidualSequence,
    compute_energy_concentration,
    compute_per_dimension_metrics,
    compute_prediction_residual,
    normalize_residual,
    reconstruct_motion,
    retain_random_k,
    retain_top_k,
    selection_accounting,
)
from av_semcom.models.residual.config import ResidualSettings


def run_residual_analysis(
    settings: ResidualSettings,
    predictor_settings: AudioMotionSettings,
    samples: Sequence[GridSample],
    e3_run_dir: Path,
    *,
    run_directory: Path | None = None,
    resume: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Analyze frozen E3 prediction residuals without retraining or rendering."""

    e3_run_dir = e3_run_dir.resolve()
    e3_experiment = _load_complete_e3(e3_run_dir)
    e3_fingerprint = str(e3_experiment["experiment_fingerprint"])
    selected_seed = _best_validation_seed(e3_run_dir)
    evaluation_samples = sorted(
        (sample for sample in samples if sample.split in predictor_settings.evaluation_splits),
        key=lambda sample: (sample.split, sample.sample_id),
    )
    if not evaluation_samples:
        raise ValueError("residual analysis has no validation/test samples")
    _validate_e3_reconstruction(e3_run_dir, selected_seed, len(evaluation_samples))
    prediction_hashes = _prediction_hashes(
        evaluation_samples,
        e3_run_dir,
        selected_seed,
    )
    inputs = {
        "e3_experiment_fingerprint": e3_fingerprint,
        "selected_seed": selected_seed,
        "seed_selection": "minimum_validation_l1_only",
        "prediction_count": len(prediction_hashes),
        "prediction_tree_sha256": config_fingerprint(prediction_hashes),
        "motion_stats_sha256": file_sha256(predictor_settings.motion_stats_path),
    }
    experiment_fingerprint = config_fingerprint(
        {
            "config": settings.config,
            "inputs": inputs,
        }
    )
    run_dir = run_directory or _new_run_directory(settings.output_root)
    _prepare_run_directory(
        run_dir,
        settings,
        inputs,
        prediction_hashes,
        experiment_fingerprint,
        resume=resume,
    )
    complete_path = run_dir / "analysis_complete.json"
    if resume and complete_path.is_file():
        _require_fingerprint(complete_path, experiment_fingerprint)
        return run_dir, _read_json(run_dir / "summary.json")

    normalizer = load_motion_normalizer(predictor_settings.motion_stats_path)
    if normalizer.scope != "train_stats":
        raise ValueError("E4 requires frozen train_stats motion normalization")
    sample_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    for position, sample in enumerate(evaluation_samples, start=1):
        print(
            f"[residual-analysis] sample {position}/{len(evaluation_samples)}: {sample.sample_id}",
            flush=True,
        )
        sample_path = run_dir / "samples" / f"{sample.sample_id}.json"
        if sample_path.is_file():
            if not resume:
                raise FileExistsError(f"residual sample result already exists: {sample_path}")
            payload = _read_json(sample_path)
            if payload.get("experiment_fingerprint") != experiment_fingerprint:
                raise ValueError(f"stale residual sample result: {sample_path}")
            if len(payload.get("selection_rows", [])) != _expected_rows_per_sample(settings):
                raise ValueError(f"incomplete residual sample result: {sample_path}")
            sample_rows.extend(payload["selection_rows"])
            residual_rows.append(payload["residual_statistics"])
            continue
        payload = _analyze_sample(
            sample,
            e3_run_dir,
            e3_fingerprint,
            selected_seed,
            normalizer.std,
            settings,
        )
        atomic_write_json(
            sample_path,
            {
                "experiment_fingerprint": experiment_fingerprint,
                **payload,
            },
        )
        sample_rows.extend(payload["selection_rows"])
        residual_rows.append(payload["residual_statistics"])

    _atomic_write_jsonl(run_dir / "selection_metrics.jsonl", sample_rows)
    _atomic_write_jsonl(run_dir / "residual_statistics.jsonl", residual_rows)
    summary = _summarize(sample_rows, residual_rows)
    atomic_write_json(run_dir / "summary.json", summary)
    _write_summary_csv(run_dir / "summary.csv", summary)
    _write_plots(run_dir / "plots", summary)
    atomic_write_json(
        complete_path,
        {
            "experiment_fingerprint": experiment_fingerprint,
            "sample_count": len(evaluation_samples),
            "selection_result_count": len(sample_rows),
            "selected_seed": selected_seed,
        },
    )
    atomic_write_json(
        run_dir / "experiment.json",
        {
            "experiment_fingerprint": experiment_fingerprint,
            "e3_experiment_fingerprint": e3_fingerprint,
            "selected_seed": selected_seed,
            "status": "analysis_complete",
            "sample_count": len(evaluation_samples),
            "selection_result_count": len(sample_rows),
        },
    )
    return run_dir, summary


def _analyze_sample(
    sample: GridSample,
    e3_run_dir: Path,
    e3_fingerprint: str,
    selected_seed: int,
    motion_std: np.ndarray,
    settings: ResidualSettings,
) -> dict[str, Any]:
    prediction_path = (
        e3_run_dir
        / f"seed_{selected_seed}"
        / "predictions"
        / sample.split
        / f"{sample.sample_id}.npz"
    )
    artifact = load_prediction(
        prediction_path,
        expected_fingerprint=e3_fingerprint,
    )
    _validate_prediction_artifact(artifact, sample, selected_seed)
    prediction = artifact["prediction"]
    target = artifact["target"]
    valid_mask = artifact["valid_mask"]
    original = compute_prediction_residual(target, prediction, valid_mask)
    normalized = normalize_residual(original, motion_std)
    metrics = compute_per_dimension_metrics(original, normalized)
    concentration = {
        "raw": compute_energy_concentration(original).to_dict(),
        "normalized": compute_energy_concentration(normalized).to_dict(),
    }
    rows = [
        _dense_row(
            sample,
            condition="dense_motion_oracle",
            target=target,
            candidate=target,
            valid_mask=valid_mask,
            eligible_frames=int(original.transmission_mask.sum()),
            settings=settings,
        )
    ]
    for k in settings.budgets:
        if k == 0:
            selection = retain_top_k(original, 0)
            rows.append(
                _selection_row(
                    sample,
                    condition="prediction_only",
                    selection_space="none",
                    selection=selection,
                    original_residual=original,
                    normalized_residual=normalized,
                    prediction=prediction,
                    target=target,
                    valid_mask=valid_mask,
                    settings=settings,
                )
            )
            continue
        if k == 18:
            selection = retain_top_k(original, 18)
            rows.append(
                _selection_row(
                    sample,
                    condition="full_residual_oracle",
                    selection_space="raw",
                    selection=selection,
                    original_residual=original,
                    normalized_residual=normalized,
                    prediction=prediction,
                    target=target,
                    valid_mask=valid_mask,
                    settings=settings,
                )
            )
            continue
        for selection_space in settings.selection_spaces:
            selection = retain_top_k(
                original,
                k,
                scores=normalized if selection_space == "normalized" else None,
            )
            rows.append(
                _selection_row(
                    sample,
                    condition="magnitude_top_k",
                    selection_space=selection_space,
                    selection=selection,
                    original_residual=original,
                    normalized_residual=normalized,
                    prediction=prediction,
                    target=target,
                    valid_mask=valid_mask,
                    settings=settings,
                )
            )
        for base_seed in settings.random_seeds:
            derived_seed = _sample_seed(base_seed, sample.sample_id)
            selection = retain_random_k(original, k, seed=derived_seed)
            rows.append(
                _selection_row(
                    sample,
                    condition="random_k",
                    selection_space="none",
                    selection=selection,
                    original_residual=original,
                    normalized_residual=normalized,
                    prediction=prediction,
                    target=target,
                    valid_mask=valid_mask,
                    settings=settings,
                    reported_seed=base_seed,
                )
            )
    return {
        "sample_id": sample.sample_id,
        "speaker_id": sample.speaker_id,
        "split": sample.split,
        "selected_seed": selected_seed,
        "prediction_sha256": file_sha256(prediction_path),
        "residual_statistics": {
            "sample_id": sample.sample_id,
            "speaker_id": sample.speaker_id,
            "split": sample.split,
            "per_dimension": metrics.to_dict(),
            "concentration": concentration,
        },
        "selection_rows": rows,
    }


def _selection_row(
    sample: GridSample,
    *,
    condition: str,
    selection_space: str,
    selection: ResidualSelection,
    original_residual: ResidualSequence,
    normalized_residual: ResidualSequence,
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    settings: ResidualSettings,
    reported_seed: int | None = None,
) -> dict[str, Any]:
    candidate = reconstruct_motion(prediction, selection)
    motion = _masked_motion_metrics(target, candidate, valid_mask)
    retained_normalized = np.where(
        selection.selection_mask,
        normalized_residual.values,
        0.0,
    )
    original_energy = float(np.square(original_residual.values, dtype=np.float64).sum())
    normalized_energy = float(np.square(normalized_residual.values, dtype=np.float64).sum())
    accounting = selection_accounting(selection)
    adaptive_indices = condition == "magnitude_top_k" and 0 < selection.k < 18
    index_count = accounting.retained_value_count if adaptive_indices else 0
    index_bits = index_count * settings.dimension_index_bits
    combination_bits_per_frame = (
        math.ceil(math.log2(math.comb(18, selection.k))) if adaptive_indices else 0
    )
    combination_index_bits = accounting.eligible_frame_count * combination_bits_per_frame
    value_bits = accounting.retained_value_count * settings.value_storage_bits
    return {
        "sample_id": sample.sample_id,
        "speaker_id": sample.speaker_id,
        "split": sample.split,
        "condition": condition,
        "selection_space": selection_space,
        "k": selection.k,
        "seed": reported_seed,
        **motion,
        "raw_energy_retained_fraction": _energy_fraction(
            selection.retained.values,
            original_energy,
        ),
        "normalized_energy_retained_fraction": _energy_fraction(
            retained_normalized,
            normalized_energy,
        ),
        **accounting.to_dict(),
        "dimension_index_count": index_count,
        "dimension_index_bit_count": index_bits,
        "combination_index_bits_per_frame_lower_bound": combination_bits_per_frame,
        "combination_index_bit_count_lower_bound": combination_index_bits,
        "index_scheme": (
            "adaptive_top_k_indices"
            if adaptive_indices
            else "shared_random_schedule"
            if condition == "random_k"
            else "no_indices"
        ),
        "value_storage_bits_assumption": settings.value_storage_bits,
        "value_storage_bit_count": value_bits,
        "uncoded_payload_bit_count_proxy": value_bits + index_bits,
        "combination_lower_bound_payload_bit_count_proxy": value_bits + combination_index_bits,
        "rate_scope": "residual_values_and_fixed_width_dimension_indices_only",
    }


def _dense_row(
    sample: GridSample,
    *,
    condition: str,
    target: np.ndarray,
    candidate: np.ndarray,
    valid_mask: np.ndarray,
    eligible_frames: int,
    settings: ResidualSettings,
) -> dict[str, Any]:
    motion = _masked_motion_metrics(target, candidate, valid_mask)
    value_count = eligible_frames * 18
    return {
        "sample_id": sample.sample_id,
        "speaker_id": sample.speaker_id,
        "split": sample.split,
        "condition": condition,
        "selection_space": "none",
        "k": 18,
        "seed": None,
        **motion,
        "raw_energy_retained_fraction": 1.0,
        "normalized_energy_retained_fraction": 1.0,
        "frame_count": int(target.shape[0]),
        "eligible_frame_count": eligible_frames,
        "dimensions_per_frame": 18,
        "dense_value_count": value_count,
        "retained_value_count": value_count,
        "dimension_index_count": 0,
        "dimension_index_bits_per_value": settings.dimension_index_bits,
        "dimension_index_bit_count": 0,
        "combination_index_bits_per_frame_lower_bound": 0,
        "combination_index_bit_count_lower_bound": 0,
        "index_scheme": "no_indices",
        "accounting_scope": "scalar_and_fixed_width_dimension_index_counts_only",
        "value_storage_bits_assumption": settings.value_storage_bits,
        "value_storage_bit_count": value_count * settings.value_storage_bits,
        "uncoded_payload_bit_count_proxy": value_count * settings.value_storage_bits,
        "combination_lower_bound_payload_bit_count_proxy": value_count
        * settings.value_storage_bits,
        "rate_scope": "motion_values_only",
    }


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    residual_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, int, int | None], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        key = (
            str(row["split"]),
            str(row["condition"]),
            str(row["selection_space"]),
            int(row["k"]),
            row.get("seed"),
        )
        grouped[key].append(row)
    groups: list[dict[str, Any]] = []
    mean_fields = (
        "l1",
        "rmse",
        "velocity_l1",
        "raw_energy_retained_fraction",
        "normalized_energy_retained_fraction",
        "retained_value_count",
        "dimension_index_bit_count",
        "uncoded_payload_bit_count_proxy",
    )
    for (split, condition, space, k, seed), members in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][3],
            item[0][1],
            item[0][2],
            -1 if item[0][4] is None else item[0][4],
        ),
    ):
        groups.append(
            {
                "split": split,
                "condition": condition,
                "selection_space": space,
                "k": k,
                "seed": seed,
                "sample_count": len(members),
                **{
                    field: float(np.mean([float(row[field]) for row in members]))
                    for field in mean_fields
                },
            }
        )
    random_aggregate: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        for k in sorted({int(group["k"]) for group in groups if group["condition"] == "random_k"}):
            members = [
                group
                for group in groups
                if group["split"] == split and group["condition"] == "random_k" and group["k"] == k
            ]
            if not members:
                continue
            random_aggregate.append(
                {
                    "split": split,
                    "condition": "random_k",
                    "k": k,
                    "seed_count": len(members),
                    **{
                        f"{metric}_{suffix}": float(
                            function([float(group[metric]) for group in members])
                        )
                        for metric in ("l1", "rmse", "velocity_l1")
                        for suffix, function in (("mean", np.mean), ("std", np.std))
                    },
                }
            )
    residual_summary = _summarize_residual_statistics(residual_rows)
    return {
        "schema_version": 1,
        "selection_result_count": len(rows),
        "sample_count": len(residual_rows),
        "groups": groups,
        "random_seed_aggregate": random_aggregate,
        "residual_statistics": residual_summary,
        "rate_definition": {
            "primary_axis": "retained_residual_values_per_frame_k",
            "dimension_index_bits_per_retained_value": 5,
            "value_storage_bits_assumption": 32,
            "uncoded_payload_proxy_excludes": [
                "audio",
                "reference_image",
                "entropy_coding",
                "packet_headers",
                "channel_coding",
            ],
        },
    }


def _summarize_residual_statistics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("validation", "test"):
        members = [row for row in rows if row["split"] == split]
        if not members:
            continue
        split_result: dict[str, Any] = {}
        for space in ("original", "normalized"):
            split_result[space] = {
                metric: np.mean(
                    [
                        np.asarray(row["per_dimension"][space][metric], dtype=np.float64)
                        for row in members
                    ],
                    axis=0,
                ).tolist()
                for metric in ("l1", "rmse", "velocity_l1")
            }
        for space in ("raw", "normalized"):
            dimension_curves = [
                np.asarray(
                    row["concentration"][space]["dimensions"]["cumulative_fraction"],
                    dtype=np.float64,
                )
                for row in members
            ]
            frame_curves = [
                np.asarray(
                    row["concentration"][space]["frames"]["cumulative_fraction"],
                    dtype=np.float64,
                )
                for row in members
            ]
            split_result[f"{space}_concentration"] = {
                "mean_dimension_cumulative_fraction_by_rank": np.mean(
                    dimension_curves,
                    axis=0,
                ).tolist(),
                "mean_frame_cumulative_fraction_by_rank": np.mean(
                    frame_curves,
                    axis=0,
                ).tolist(),
            }
        result[split] = split_result
    return result


def _prepare_run_directory(
    run_dir: Path,
    settings: ResidualSettings,
    inputs: Mapping[str, Any],
    prediction_hashes: Mapping[str, str],
    fingerprint: str,
    *,
    resume: bool,
) -> None:
    metadata_path = run_dir / "run_metadata.json"
    if run_dir.exists():
        if not resume:
            raise FileExistsError(f"run directory already exists: {run_dir}")
        metadata = _read_json(metadata_path)
        if metadata.get("experiment_fingerprint") != fingerprint:
            raise ValueError("residual run fingerprint does not match current inputs")
        return
    if resume:
        raise FileNotFoundError(f"cannot resume missing residual run: {run_dir}")
    run_dir.mkdir(parents=True)
    atomic_write_json(run_dir / "resolved_config.json", dict(settings.config))
    atomic_write_json(
        run_dir / "input_hashes.json",
        {
            **dict(inputs),
            "predictions": dict(prediction_hashes),
        },
    )
    atomic_write_json(
        metadata_path,
        {
            "experiment_fingerprint": fingerprint,
            "git_commit": _git_commit(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    atomic_write_json(run_dir / "environment.json", _environment())
    atomic_write_json(
        run_dir / "experiment.json",
        {
            "experiment_fingerprint": fingerprint,
            "status": "running",
        },
    )


def _prediction_hashes(
    samples: Sequence[GridSample],
    e3_run_dir: Path,
    selected_seed: int,
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for sample in samples:
        path = (
            e3_run_dir
            / f"seed_{selected_seed}"
            / "predictions"
            / sample.split
            / f"{sample.sample_id}.npz"
        )
        hashes[str(path.relative_to(e3_run_dir))] = file_sha256(path)
    return hashes


def _best_validation_seed(e3_run_dir: Path) -> int:
    summary = _read_json(e3_run_dir / "summary.json")
    candidates = [
        group
        for group in summary.get("groups", [])
        if group.get("method") == "audio_gru" and group.get("split") == "validation"
    ]
    if not candidates:
        raise ValueError("E3 summary contains no validation GRU groups")
    return int(min(candidates, key=lambda group: float(group["l1"]))["seed"])


def _load_complete_e3(e3_run_dir: Path) -> dict[str, Any]:
    experiment = _read_json(e3_run_dir / "experiment.json")
    if experiment.get("status") != "complete":
        raise ValueError("E3 experiment must be complete before residual analysis")
    if not experiment.get("experiment_fingerprint"):
        raise ValueError("E3 experiment has no fingerprint")
    validation = _read_json(e3_run_dir / "validation_report.json")
    if validation.get("error_count") != 0:
        raise ValueError("E3 validation report contains errors")
    return experiment


def _validate_e3_reconstruction(
    e3_run_dir: Path,
    selected_seed: int,
    expected_samples: int,
) -> None:
    marker = _read_json(e3_run_dir / "reconstruction" / "complete.json")
    if marker.get("best_validation_seed") != selected_seed:
        raise ValueError("E3 reconstruction seed does not match validation-only selection")
    if marker.get("sample_count") != expected_samples:
        raise ValueError("E3 reconstruction sample count does not match E4 inputs")
    summary = _read_json(e3_run_dir / "reconstruction" / "summary.json")
    if summary.get("failure_count") != 0:
        raise ValueError("E3 reconstruction contains failures")


def _validate_prediction_artifact(
    artifact: Mapping[str, Any],
    sample: GridSample,
    selected_seed: int,
) -> None:
    if artifact["sample_id"] != sample.sample_id:
        raise ValueError("prediction sample_id does not match manifest")
    if artifact["speaker_id"] != sample.speaker_id or artifact["split"] != sample.split:
        raise ValueError("prediction identity or split does not match manifest")
    if artifact["method"] != "audio_gru" or artifact["seed"] != selected_seed:
        raise ValueError("prediction does not belong to the selected GRU seed")


def _energy_fraction(values: np.ndarray, total_energy: float) -> float:
    retained = float(np.square(values, dtype=np.float64).sum())
    return retained / total_energy if total_energy > 0 else 1.0


def _masked_motion_metrics(
    target: np.ndarray,
    candidate: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float]:
    if target.shape != candidate.shape or target.ndim != 2 or target.shape[1] != 18:
        raise ValueError("target and candidate must share shape [T,18]")
    if valid_mask.shape != (target.shape[0],) or valid_mask.dtype != np.bool_:
        raise ValueError("valid_mask must be boolean with shape [T]")
    difference = candidate.astype(np.float64) - target.astype(np.float64)
    valid_difference = difference[valid_mask]
    if not valid_difference.size:
        raise ValueError("motion metrics require at least one valid frame")
    velocity_pairs = valid_mask[1:] & valid_mask[:-1]
    velocity_difference = np.diff(difference, axis=0)[velocity_pairs]
    return {
        "l1": float(np.abs(valid_difference).mean()),
        "rmse": float(np.sqrt(np.square(valid_difference).mean())),
        "velocity_l1": (
            float(np.abs(velocity_difference).mean()) if velocity_difference.size else 0.0
        ),
    }


def _expected_rows_per_sample(settings: ResidualSettings) -> int:
    intermediate = sum(k not in {0, 18} for k in settings.budgets)
    return 3 + intermediate * (len(settings.selection_spaces) + len(settings.random_seeds))


def _sample_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _new_run_directory(root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return root / timestamp


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    return payload


def _require_fingerprint(path: Path, fingerprint: str) -> None:
    if _read_json(path).get("experiment_fingerprint") != fingerprint:
        raise ValueError(f"artifact fingerprint does not match: {path}")


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


def _write_summary_csv(path: Path, summary: Mapping[str, Any]) -> None:
    groups = summary["groups"]
    if not groups:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)


def _write_plots(root: Path, summary: Mapping[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    root.mkdir(parents=True, exist_ok=True)
    for split in ("validation", "test"):
        figure, axis = plt.subplots(figsize=(8, 5))
        for space, label in (("raw", "raw magnitude"), ("normalized", "normalized magnitude")):
            groups = [
                group
                for group in summary["groups"]
                if group["split"] == split
                and group["condition"] == "magnitude_top_k"
                and group["selection_space"] == space
            ]
            axis.plot(
                [group["k"] for group in groups],
                [group["l1"] for group in groups],
                marker="o",
                label=label,
            )
        random_groups = [
            group for group in summary["random_seed_aggregate"] if group["split"] == split
        ]
        axis.plot(
            [group["k"] for group in random_groups],
            [group["l1_mean"] for group in random_groups],
            marker="o",
            label="random mean",
        )
        axis.set_xlabel("Retained residual values per frame (k of 18)")
        axis.set_ylabel("Raw motion L1")
        axis.set_title(f"E4 residual retention: {split}")
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(root / f"rate_quality_{split}.png", dpi=160)
        plt.close(figure)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "unknown"


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
