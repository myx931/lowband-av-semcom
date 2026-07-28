"""Frozen LivePortrait evaluation of exported Sionna JSCC motion candidates."""

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
from av_semcom.models.jscc.candidates import (
    JSCCCandidateBundle,
    JSCCCondition,
    load_candidate_bundle,
)
from av_semcom.models.jscc.config import JSCCReconstructionSettings, JSCCSettings
from av_semcom.models.motion.config import MotionSettings
from av_semcom.models.motion.pipeline import create_reconstruction_backend
from av_semcom.models.motion.sequence import load_motion_sequence
from av_semcom.models.predictor.reconstruction import (
    _save_contact_sheet,
    _save_video,
    _ThreadLocalMediaPipeFaceMeshBackend,
)
from av_semcom.models.reconstruction.backend import ReconstructionBackend


def run_jscc_reconstruction(
    settings: JSCCSettings,
    reconstruction: JSCCReconstructionSettings,
    motion_settings: MotionSettings,
    samples: Sequence[GridSample],
    run_dir: Path,
    *,
    resume: bool = False,
    backend: ReconstructionBackend | None = None,
    landmark_backend: FaceLandmarkBackend | None = None,
) -> tuple[dict[str, Any], list[FailureRecord]]:
    """Render and score every frozen E5 video condition."""

    run_dir = run_dir.resolve()
    candidate_complete = _read_json(run_dir / "reconstruction_candidates" / "complete.json")
    if candidate_complete.get("status") != "complete":
        raise ValueError("JSCC reconstruction candidates are not complete")
    candidate_fingerprint = str(candidate_complete.get("candidate_fingerprint", ""))
    experiment_fingerprint = str(candidate_complete.get("experiment_fingerprint", ""))
    if not candidate_fingerprint or not experiment_fingerprint:
        raise ValueError("candidate completion marker is missing fingerprints")
    evaluation_samples = sorted(
        (sample for sample in samples if sample.split == reconstruction.split),
        key=lambda sample: sample.sample_id,
    )
    if len(evaluation_samples) != int(candidate_complete["sample_count"]):
        raise ValueError("candidate and manifest sample counts differ")
    condition_count = int(candidate_complete["condition_count_per_sample"])
    output_root = run_dir / "video_reconstruction"
    runtime = {
        "experiment_fingerprint": experiment_fingerprint,
        "candidate_fingerprint": candidate_fingerprint,
        "sample_count": len(evaluation_samples),
        "condition_count_per_sample": condition_count,
        "split": reconstruction.split,
        "noise_seed": reconstruction.noise_seed,
        "backend": motion_settings.backend,
        "backend_revision": motion_settings.backend_revision,
        "reconstruction_batch_size": motion_settings.reconstruction_batch_size,
        "metric_workers": reconstruction.metric_workers,
        "media_channel_uses": reconstruction.media_channel_uses,
    }
    runtime_path = output_root / "runtime.json"
    if runtime_path.is_file():
        if _read_json(runtime_path) != runtime:
            raise ValueError("JSCC reconstruction runtime settings do not match")
    else:
        atomic_write_json(runtime_path, runtime)
    complete_path = output_root / "complete.json"
    if complete_path.is_file() and not resume:
        raise FileExistsError("JSCC video reconstruction is complete; pass --resume")
    if resume and complete_path.is_file():
        complete = _read_json(complete_path)
        if complete.get("candidate_fingerprint") != candidate_fingerprint:
            raise ValueError("video reconstruction completion fingerprint mismatch")
        return _read_json(output_root / "summary.json"), []

    representative_ids = _representative_sample_ids(evaluation_samples)
    owns_backend = backend is None
    active_backend = backend or create_reconstruction_backend(motion_settings)
    owns_landmarks = landmark_backend is None
    parallel_metrics = reconstruction.metric_workers > 1 and landmark_backend is None
    landmarks: FaceLandmarkBackend = landmark_backend or (
        _ThreadLocalMediaPipeFaceMeshBackend() if parallel_metrics else MediaPipeFaceMeshBackend()
    )
    metric_executor = (
        ThreadPoolExecutor(
            max_workers=reconstruction.metric_workers,
            thread_name_prefix="jscc-reconstruction-metrics",
        )
        if parallel_metrics
        else None
    )
    rows: list[dict[str, Any]] = []
    failures: list[FailureRecord] = []
    try:
        for position, sample in enumerate(evaluation_samples, start=1):
            print(
                f"[jscc-reconstruction] sample {position}/{len(evaluation_samples)}: "
                f"{sample.sample_id}",
                flush=True,
            )
            sample_path = output_root / "samples" / f"{sample.sample_id}.json"
            if sample_path.is_file():
                if not resume:
                    raise FileExistsError(
                        f"JSCC reconstruction result already exists: {sample_path}"
                    )
                payload = _read_json(sample_path)
                if payload.get("candidate_fingerprint") != candidate_fingerprint:
                    raise ValueError(f"stale JSCC reconstruction result: {sample_path}")
                if len(payload.get("rows", [])) != condition_count:
                    raise ValueError(f"incomplete JSCC reconstruction result: {sample_path}")
                rows.extend(payload["rows"])
                continue
            try:
                sample_rows = _evaluate_sample(
                    sample,
                    reconstruction,
                    motion_settings,
                    active_backend,
                    landmarks,
                    metric_executor,
                    run_dir,
                    candidate_fingerprint,
                    output_root,
                    save_media=(
                        reconstruction.save_representative_media
                        and sample.sample_id in representative_ids
                    ),
                )
                if len(sample_rows) != condition_count:
                    raise RuntimeError("renderer returned an unexpected condition count")
                atomic_write_json(
                    sample_path,
                    {
                        "experiment_fingerprint": experiment_fingerprint,
                        "candidate_fingerprint": candidate_fingerprint,
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
                        stage="jscc_video_reconstruction",
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
    _write_plots(output_root / "plots", summary)
    if not failures and len(rows) == len(evaluation_samples) * condition_count:
        atomic_write_json(
            complete_path,
            {
                "experiment_fingerprint": experiment_fingerprint,
                "candidate_fingerprint": candidate_fingerprint,
                "sample_count": len(evaluation_samples),
                "result_count": len(rows),
                "failure_count": 0,
                "status": "complete",
            },
        )
    return summary, failures


def _evaluate_sample(
    sample: GridSample,
    reconstruction: JSCCReconstructionSettings,
    motion_settings: MotionSettings,
    backend: ReconstructionBackend,
    landmarks: FaceLandmarkBackend,
    metric_executor: ThreadPoolExecutor | None,
    run_dir: Path,
    candidate_fingerprint: str,
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
        resolve_record_path(
            sample.face_crop_path,
            motion_settings.data_settings.data_root,
        ),
        allow_pickle=False,
    ) as payload:
        original_frames = payload["crops"].astype(np.uint8)
    bundle = load_candidate_bundle(
        run_dir / "reconstruction_candidates" / reconstruction.split / f"{sample.sample_id}.npz",
        expected_fingerprint=candidate_fingerprint,
    )
    if (
        bundle.sample_id != sample.sample_id
        or bundle.speaker_id != sample.speaker_id
        or bundle.split != sample.split
    ):
        raise ValueError("candidate identity does not match manifest")
    if not np.array_equal(bundle.valid_mask, sequence.valid_mask):
        raise ValueError("candidate and motion validity masks differ")
    source = original_frames[sequence.source_frame_index]
    reconstructed_sets = backend.reconstruct_lip_vectors(
        source,
        sequence,
        [vector for vector in bundle.vectors],
    )
    if len(reconstructed_sets) != len(bundle.conditions):
        raise RuntimeError("reconstruction backend returned the wrong candidate count")
    oracle_index = next(
        index
        for index, condition in enumerate(bundle.conditions)
        if condition.family == "full_residual_oracle"
    )
    oracle_frames = reconstructed_sets[oracle_index]
    original_detections = _detect_sequence(landmarks, original_frames)
    oracle_detections = _detect_sequence(landmarks, oracle_frames)
    rows: list[dict[str, Any]] = []
    jobs: list[Future[dict[str, Any]]] = []
    for condition, reconstructed in zip(
        bundle.conditions,
        reconstructed_sets,
        strict=True,
    ):
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
    rows.extend(job.result() for job in jobs)
    if save_media:
        _save_representative_media(
            output_root,
            sample,
            sequence.fps,
            original_frames,
            oracle_frames,
            bundle,
            reconstructed_sets,
            media_channel_uses=reconstruction.media_channel_uses,
        )
    return rows


def _metric_row(
    sample: GridSample,
    condition: JSCCCondition,
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
    return {
        "sample_id": sample.sample_id,
        "speaker_id": sample.speaker_id,
        "split": sample.split,
        **condition.to_dict(),
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
    if not samples:
        return set()
    return {
        samples[0].sample_id,
        samples[len(samples) // 2].sample_id,
        samples[-1].sample_id,
    }


def _save_representative_media(
    root: Path,
    sample: GridSample,
    fps: float,
    original: np.ndarray,
    oracle: np.ndarray,
    bundle: JSCCCandidateBundle,
    reconstructed_sets: Sequence[np.ndarray],
    *,
    media_channel_uses: int,
) -> None:
    media_root = root / "media" / sample.split / sample.sample_id
    _save_video(media_root / "original.mp4", original, fps)
    _save_video(media_root / "oracle.mp4", oracle, fps)
    for condition, frames in zip(bundle.conditions, reconstructed_sets, strict=True):
        selected = condition.family == "prediction_only" or (
            condition.channel_uses == media_channel_uses
            and condition.family in {"noiseless_autoencoder", "jscc_awgn"}
        )
        if not selected:
            continue
        _save_video(media_root / f"{condition.condition_id}.mp4", frames, fps)
        _save_contact_sheet(
            media_root / f"{condition.condition_id}.jpg",
            original,
            oracle,
            frames,
        )


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    failures: Sequence[FailureRecord],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition_id"])].append(row)
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
    for condition_id_value, members in sorted(grouped.items()):
        first = members[0]
        group: dict[str, Any] = {
            "condition_id": condition_id_value,
            "family": first["family"],
            "channel_uses": first.get("channel_uses"),
            "model_seed": first.get("model_seed"),
            "snr_db": first.get("snr_db"),
            "noise_seed": first.get("noise_seed"),
            "split": first["split"],
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


def _write_plots(path: Path, summary: Mapping[str, Any]) -> None:
    groups = summary["groups"]
    prediction = next(
        (group for group in groups if group["family"] == "prediction_only"),
        None,
    )
    if prediction is None:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    path.mkdir(parents=True, exist_ok=True)
    for metric, label, filename in (
        ("oracle_mouth_mae", "Oracle-relative mouth ROI MAE", "mouth_mae_vs_snr.png"),
        ("oracle_mouth_nme", "Oracle-relative mouth NME", "mouth_nme_vs_snr.png"),
        ("oracle_ssim", "Oracle-relative SSIM", "ssim_vs_snr.png"),
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.axhline(
            float(prediction[metric]),
            color="black",
            linestyle="--",
            label="prediction only",
        )
        channel_uses = sorted(
            {int(group["channel_uses"]) for group in groups if group["family"] == "jscc_awgn"}
        )
        for uses in channel_uses:
            members = sorted(
                (
                    group
                    for group in groups
                    if group["family"] == "jscc_awgn" and group["channel_uses"] == uses
                ),
                key=lambda group: float(group["snr_db"]),
            )
            axis.plot(
                [group["snr_db"] for group in members],
                [group[metric] for group in members],
                marker="o",
                label=f"C={uses}",
            )
        axis.set_xlabel("SNR (dB)")
        axis.set_ylabel(label)
        axis.set_title(f"Sionna JSCC LivePortrait: {label}")
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(path / filename, dpi=160)
        plt.close(figure)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    return payload
