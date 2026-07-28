"""Frozen-renderer evaluation of retained prediction residuals."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from av_semcom.data.grid import GridSample, resolve_record_path
from av_semcom.data.landmarks import FaceLandmarkBackend, MediaPipeFaceMeshBackend
from av_semcom.data.preprocessing import FailureRecord, atomic_write_json, write_failures
from av_semcom.metrics.motion import compute_reconstruction_metrics
from av_semcom.models.motion.config import MotionSettings
from av_semcom.models.motion.perturbations import load_motion_normalizer
from av_semcom.models.motion.pipeline import create_reconstruction_backend
from av_semcom.models.motion.sequence import load_motion_sequence
from av_semcom.models.predictor.artifacts import load_prediction
from av_semcom.models.predictor.reconstruction import (
    _save_contact_sheet,
    _save_video,
    _ThreadLocalMediaPipeFaceMeshBackend,
)
from av_semcom.models.reconstruction.backend import ReconstructionBackend
from av_semcom.models.residual.analysis import (
    ResidualSequence,
    compute_prediction_residual,
    normalize_residual,
    reconstruct_motion,
    retain_random_k,
    retain_top_k,
)
from av_semcom.models.residual.config import ResidualSettings
from av_semcom.models.residual.experiment import _sample_seed


def run_residual_reconstruction(
    settings: ResidualSettings,
    motion_settings: MotionSettings,
    samples: Sequence[GridSample],
    e3_run_dir: Path,
    residual_run_dir: Path,
    *,
    resume: bool = False,
    backend: ReconstructionBackend | None = None,
    landmark_backend: FaceLandmarkBackend | None = None,
    save_representative_media: bool = True,
) -> tuple[dict[str, Any], list[FailureRecord]]:
    """Render all configured retained-residual candidates and score them."""

    experiment = _read_json(residual_run_dir / "experiment.json")
    if experiment.get("status") not in {"analysis_complete", "complete"}:
        raise ValueError("residual motion analysis is not complete")
    fingerprint = str(experiment.get("experiment_fingerprint", ""))
    e3_fingerprint = str(experiment.get("e3_experiment_fingerprint", ""))
    selected_seed = int(experiment["selected_seed"])
    if not fingerprint or not e3_fingerprint:
        raise ValueError("residual experiment is missing input fingerprints")
    evaluation_samples = sorted(
        (sample for sample in samples if sample.split in {"validation", "test"}),
        key=lambda sample: (sample.split, sample.sample_id),
    )
    if not evaluation_samples:
        raise ValueError("residual reconstruction has no validation/test samples")
    output_root = residual_run_dir / "reconstruction"
    runtime = {
        "experiment_fingerprint": fingerprint,
        "sample_count": len(evaluation_samples),
        "reconstruction_budgets": list(settings.reconstruction_budgets),
        "random_seeds": list(settings.random_seeds),
        "selection_spaces": list(settings.selection_spaces),
        "reconstruction_batch_size": motion_settings.reconstruction_batch_size,
        "metric_workers": settings.metric_workers,
    }
    runtime_path = output_root / "runtime.json"
    if runtime_path.is_file():
        if _read_json(runtime_path) != runtime:
            raise ValueError("residual reconstruction runtime settings do not match")
    else:
        atomic_write_json(runtime_path, runtime)
    complete_path = output_root / "complete.json"
    if resume and complete_path.is_file():
        _require_fingerprint(complete_path, fingerprint)
        return _read_json(output_root / "summary.json"), []

    representative_ids = _representative_sample_ids(evaluation_samples)
    normalizer = load_motion_normalizer(motion_settings.stats_path)
    if normalizer.scope != "train_stats":
        raise ValueError("residual reconstruction requires train_stats")
    owns_backend = backend is None
    active_backend = backend or create_reconstruction_backend(motion_settings)
    owns_landmarks = landmark_backend is None
    parallel_metrics = settings.metric_workers > 1 and landmark_backend is None
    landmarks: FaceLandmarkBackend = landmark_backend or (
        _ThreadLocalMediaPipeFaceMeshBackend() if parallel_metrics else MediaPipeFaceMeshBackend()
    )
    metric_executor = (
        ThreadPoolExecutor(
            max_workers=settings.metric_workers,
            thread_name_prefix="residual-reconstruction-metrics",
        )
        if parallel_metrics
        else None
    )
    rows: list[dict[str, Any]] = []
    failures: list[FailureRecord] = []
    try:
        for position, sample in enumerate(evaluation_samples, start=1):
            print(
                f"[residual-reconstruction] sample {position}/{len(evaluation_samples)}: "
                f"{sample.sample_id}",
                flush=True,
            )
            sample_path = output_root / "samples" / f"{sample.sample_id}.json"
            if sample_path.is_file():
                if not resume:
                    raise FileExistsError(
                        f"residual reconstruction result already exists: {sample_path}"
                    )
                payload = _read_json(sample_path)
                if payload.get("experiment_fingerprint") != fingerprint:
                    raise ValueError(f"stale reconstruction sample: {sample_path}")
                if len(payload.get("rows", [])) != _expected_rows_per_sample(settings):
                    raise ValueError(f"incomplete reconstruction sample: {sample_path}")
                rows.extend(payload["rows"])
                continue
            try:
                sample_rows = _evaluate_sample(
                    sample,
                    settings,
                    motion_settings,
                    active_backend,
                    landmarks,
                    metric_executor,
                    e3_run_dir,
                    e3_fingerprint,
                    selected_seed,
                    normalizer.std,
                    output_root,
                    save_media=(
                        save_representative_media and sample.sample_id in representative_ids
                    ),
                )
                atomic_write_json(
                    sample_path,
                    {
                        "experiment_fingerprint": fingerprint,
                        "sample_id": sample.sample_id,
                        "rows": sample_rows,
                    },
                )
                rows.extend(sample_rows)
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                failures.append(
                    FailureRecord(
                        sample_id=sample.sample_id,
                        speaker_id=sample.speaker_id,
                        stage="residual_reconstruction",
                        reason=str(exc),
                    )
                )
    finally:
        if metric_executor is not None:
            metric_executor.shutdown(wait=True, cancel_futures=True)
        if owns_backend:
            active_backend.close()
        if owns_landmarks:
            landmarks.close()

    _atomic_write_jsonl(output_root / "metrics.jsonl", rows)
    write_failures(output_root / "failures.jsonl", failures)
    summary = _summarize(rows, failures)
    atomic_write_json(output_root / "summary.json", summary)
    _write_summary_csv(output_root / "summary.csv", summary)
    if not failures:
        atomic_write_json(
            complete_path,
            {
                "experiment_fingerprint": fingerprint,
                "sample_count": len(evaluation_samples),
                "result_count": len(rows),
                "selected_seed": selected_seed,
            },
        )
        atomic_write_json(
            residual_run_dir / "experiment.json",
            {
                **experiment,
                "status": "complete",
                "reconstruction_result_count": len(rows),
            },
        )
    return summary, failures


def _evaluate_sample(
    sample: GridSample,
    settings: ResidualSettings,
    motion_settings: MotionSettings,
    backend: ReconstructionBackend,
    landmarks: FaceLandmarkBackend,
    metric_executor: ThreadPoolExecutor | None,
    e3_run_dir: Path,
    e3_fingerprint: str,
    selected_seed: int,
    motion_std: np.ndarray,
    output_root: Path,
    *,
    save_media: bool,
) -> list[dict[str, Any]]:
    if sample.motion_path is None or sample.face_crop_path is None:
        raise ValueError("motion_path and face_crop_path are required")
    sequence = load_motion_sequence(
        resolve_record_path(sample.motion_path, motion_settings.data_settings.data_root)
    )
    with np.load(
        resolve_record_path(sample.face_crop_path, motion_settings.data_settings.data_root),
        allow_pickle=False,
    ) as payload:
        original_frames = payload["crops"].astype(np.uint8)
    prediction_artifact = load_prediction(
        e3_run_dir
        / f"seed_{selected_seed}"
        / "predictions"
        / sample.split
        / f"{sample.sample_id}.npz",
        expected_fingerprint=e3_fingerprint,
    )
    prediction = prediction_artifact["prediction"]
    target = prediction_artifact["target"]
    valid_mask = prediction_artifact["valid_mask"]
    if not np.array_equal(valid_mask, sequence.valid_mask):
        raise ValueError("prediction and motion validity masks differ")
    original_residual = compute_prediction_residual(target, prediction, valid_mask)
    normalized_residual = normalize_residual(original_residual, motion_std)
    candidates = _candidate_vectors(
        sample.sample_id,
        settings,
        prediction,
        original_residual,
        normalized_residual,
    )
    labels = [condition for condition, _ in candidates]
    vectors = [vector for _, vector in candidates]
    source = original_frames[sequence.source_frame_index]
    reconstructed_sets = backend.reconstruct_lip_vectors(source, sequence, vectors)
    if len(reconstructed_sets) != len(candidates):
        raise RuntimeError("reconstruction backend returned the wrong candidate count")
    full_index = labels.index(("full_residual_oracle", "raw", 18, None))
    oracle_frames = reconstructed_sets[full_index]
    original_detections = _detect_sequence(landmarks, original_frames)
    oracle_detections = _detect_sequence(landmarks, oracle_frames)
    jobs: list[Future[dict[str, Any]]] = []
    rows: list[dict[str, Any]] = []
    for condition, reconstructed in zip(labels, reconstructed_sets, strict=True):
        arguments = (
            sample,
            condition,
            original_frames,
            oracle_frames,
            reconstructed,
            landmarks,
            original_detections,
            oracle_detections,
        )
        if metric_executor is None:
            rows.append(_metric_row(*arguments))
        else:
            jobs.append(metric_executor.submit(_metric_row, *arguments))
    for job in jobs:
        rows.append(job.result())
    oracle_row = next(row for row in rows if row["condition"] == "full_residual_oracle")
    rows.append(
        {
            **oracle_row,
            "condition": "dense_motion_oracle",
            "selection_space": "none",
        }
    )
    if save_media:
        _save_representative_media(
            output_root,
            sample,
            sequence.fps,
            original_frames,
            oracle_frames,
            labels,
            reconstructed_sets,
        )
    return rows


def _candidate_vectors(
    sample_id: str,
    settings: ResidualSettings,
    prediction: np.ndarray,
    original_residual: ResidualSequence,
    normalized_residual: ResidualSequence,
) -> list[tuple[tuple[str, str, int, int | None], np.ndarray]]:
    candidates: list[tuple[tuple[str, str, int, int | None], np.ndarray]] = []
    for k in settings.reconstruction_budgets:
        if k == 0:
            candidates.append((("prediction_only", "none", 0, None), prediction))
            continue
        if k == 18:
            candidates.append(
                (
                    ("full_residual_oracle", "raw", 18, None),
                    reconstruct_motion(prediction, original_residual),
                )
            )
            continue
        for space in settings.selection_spaces:
            selection = retain_top_k(
                original_residual,
                k,
                scores=normalized_residual if space == "normalized" else None,
            )
            candidates.append(
                (
                    ("magnitude_top_k", space, k, None),
                    reconstruct_motion(prediction, selection),
                )
            )
        for seed in settings.random_seeds:
            selection = retain_random_k(
                original_residual,
                k,
                seed=_sample_seed(seed, sample_id),
            )
            candidates.append(
                (
                    ("random_k", "none", k, seed),
                    reconstruct_motion(prediction, selection),
                )
            )
    return candidates


def _metric_row(
    sample: GridSample,
    condition: tuple[str, str, int, int | None],
    original_frames: np.ndarray,
    oracle_frames: np.ndarray,
    reconstructed: np.ndarray,
    landmarks: FaceLandmarkBackend,
    original_detections: Sequence[Any],
    oracle_detections: Sequence[Any],
) -> dict[str, Any]:
    reconstructed_detections = _detect_sequence(landmarks, reconstructed)
    oracle_metrics = compute_reconstruction_metrics(
        oracle_frames,
        reconstructed,
        landmark_backend=landmarks,
        target_detections=oracle_detections,
        reconstructed_detections=reconstructed_detections,
    )
    original_metrics = compute_reconstruction_metrics(
        original_frames,
        reconstructed,
        landmark_backend=landmarks,
        target_detections=original_detections,
        reconstructed_detections=reconstructed_detections,
    )
    name, space, k, seed = condition
    return {
        "sample_id": sample.sample_id,
        "speaker_id": sample.speaker_id,
        "split": sample.split,
        "condition": name,
        "selection_space": space,
        "k": k,
        "seed": seed,
        **{f"oracle_{key}": value for key, value in oracle_metrics.to_dict().items()},
        **{f"original_{key}": value for key, value in original_metrics.to_dict().items()},
    }


def _detect_sequence(
    landmarks: FaceLandmarkBackend,
    frames: np.ndarray,
) -> tuple[Any, ...]:
    reset = getattr(landmarks, "reset", None)
    if callable(reset):
        reset()
    return tuple(landmarks.detect(frame) for frame in frames)


def _representative_sample_ids(samples: Sequence[GridSample]) -> set[str]:
    selected: set[str] = set()
    for split in ("validation", "test"):
        members = [sample for sample in samples if sample.split == split]
        for index in (0, len(members) // 2, len(members) - 1):
            if members:
                selected.add(members[index].sample_id)
    return selected


def _save_representative_media(
    root: Path,
    sample: GridSample,
    fps: float,
    original: np.ndarray,
    oracle: np.ndarray,
    labels: Sequence[tuple[str, str, int, int | None]],
    reconstructed_sets: Sequence[np.ndarray],
) -> None:
    media_root = root / "media" / sample.split / sample.sample_id
    _save_video(media_root / "original.mp4", original, fps)
    _save_video(media_root / "oracle.mp4", oracle, fps)
    selected = {
        ("prediction_only", "none", 0, None),
        ("magnitude_top_k", "raw", 4, None),
        ("magnitude_top_k", "normalized", 4, None),
        ("random_k", "none", 4, 42),
    }
    for label, frames in zip(labels, reconstructed_sets, strict=True):
        if label not in selected:
            continue
        name, space, k, seed = label
        stem = f"{name}_{space}_k{k}" + (f"_seed{seed}" if seed is not None else "")
        _save_video(media_root / f"{stem}.mp4", frames, fps)
        _save_contact_sheet(media_root / f"{stem}.jpg", original, oracle, frames)


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    failures: Sequence[FailureRecord],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, int, int | None], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[
            (
                str(row["split"]),
                str(row["condition"]),
                str(row["selection_space"]),
                int(row["k"]),
                row.get("seed"),
            )
        ].append(row)
    metric_names = (
        "oracle_face_mae",
        "oracle_psnr_db",
        "oracle_ssim",
        "oracle_mouth_mae",
        "oracle_mouth_nme",
        "oracle_landmark_coverage",
        "original_face_mae",
        "original_psnr_db",
        "original_ssim",
        "original_mouth_mae",
        "original_mouth_nme",
        "original_landmark_coverage",
    )
    groups: list[dict[str, Any]] = []
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
        group: dict[str, Any] = {
            "split": split,
            "condition": condition,
            "selection_space": space,
            "k": k,
            "seed": seed,
            "sample_count": len(members),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in members if row[metric] is not None]
            group[metric] = float(np.mean(values)) if values else None
        groups.append(group)
    return {
        "schema_version": 1,
        "result_count": len(rows),
        "failure_count": len(failures),
        "groups": groups,
    }


def _expected_rows_per_sample(settings: ResidualSettings) -> int:
    intermediate = sum(k not in {0, 18} for k in settings.reconstruction_budgets)
    return 3 + intermediate * (len(settings.selection_spaces) + len(settings.random_seeds))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON mapping: {path}")
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
