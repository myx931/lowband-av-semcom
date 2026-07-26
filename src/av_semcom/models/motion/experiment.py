"""LivePortrait mouth-motion reconstruction sensitivity experiment."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
from collections import defaultdict
from collections.abc import Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from av_semcom.data.grid import GridSample, read_manifest, resolve_record_path
from av_semcom.data.landmarks import (
    FaceDetection,
    FaceLandmarkBackend,
    MediaPipeFaceMeshBackend,
)
from av_semcom.data.preprocessing import FailureRecord, atomic_write_json, write_failures
from av_semcom.metrics.motion import (
    compute_motion_metrics,
    compute_reconstruction_metrics,
)
from av_semcom.models.motion.config import MotionSettings
from av_semcom.models.motion.perturbations import (
    PerturbationCondition,
    apply_perturbation,
    default_perturbation_conditions,
    load_motion_normalizer,
    perturbation_parameter_name,
)
from av_semcom.models.motion.pipeline import (
    create_reconstruction_backend,
    select_motion_samples,
)
from av_semcom.models.motion.sequence import load_motion_sequence
from av_semcom.models.reconstruction.backend import ReconstructionBackend


class _ThreadLocalMediaPipeFaceMeshBackend:
    """Give each metric worker its own stateful MediaPipe graph."""

    def __init__(self) -> None:
        self._local = threading.local()
        self._instances: list[MediaPipeFaceMeshBackend] = []
        self._lock = threading.Lock()

    def detect(self, rgb_image: np.ndarray) -> FaceDetection | None:
        backend = getattr(self._local, "backend", None)
        if backend is None:
            backend = MediaPipeFaceMeshBackend()
            self._local.backend = backend
            with self._lock:
                self._instances.append(backend)
        return backend.detect(rgb_image)

    def close(self) -> None:
        with self._lock:
            instances = tuple(self._instances)
            self._instances.clear()
        for backend in instances:
            backend.close()


def run_motion_sensitivity(
    settings: MotionSettings,
    *,
    backend: ReconstructionBackend | None = None,
    landmark_backend: FaceLandmarkBackend | None = None,
    output_directory: Path | None = None,
    conditions: Iterable[PerturbationCondition] | None = None,
) -> tuple[Path, dict[str, Any], list[FailureRecord]]:
    """Run the fixed E2 perturbation grid and save reproducible results."""

    samples = select_motion_samples(
        read_manifest(settings.data_settings.manifest_path),
        settings.data_settings,
    )
    if not samples:
        raise ValueError("motion sensitivity requires a non-empty manifest")
    normalizer = load_motion_normalizer(settings.stats_path)
    run_directory = output_directory or _new_run_directory(settings.experiment_root)
    run_directory.mkdir(parents=True, exist_ok=False)
    _write_run_metadata(run_directory, settings)

    owns_backend = backend is None
    active_backend = backend or create_reconstruction_backend(settings)
    owns_landmarks = landmark_backend is None
    parallel_metrics = settings.metric_workers > 1 and landmark_backend is None
    active_landmarks: FaceLandmarkBackend = landmark_backend or (
        _ThreadLocalMediaPipeFaceMeshBackend() if parallel_metrics else MediaPipeFaceMeshBackend()
    )
    metric_executor = (
        ThreadPoolExecutor(
            max_workers=settings.metric_workers,
            thread_name_prefix="motion-metrics",
        )
        if parallel_metrics
        else None
    )
    active_conditions = tuple(conditions or default_perturbation_conditions())
    rows: list[dict[str, Any]] = []
    failures: list[FailureRecord] = []

    try:
        if (
            active_backend.name != settings.backend
            or active_backend.revision != settings.backend_revision
        ):
            raise ValueError("active backend does not match the experiment configuration")
        for sample_position, sample in enumerate(samples):
            print(
                f"[motion-sensitivity] sample {sample_position + 1}/{len(samples)}: "
                f"{sample.sample_id}",
                file=sys.stderr,
                flush=True,
            )
            try:
                sequence = _load_sample_motion(sample, settings)
                crops, valid_mask = _load_sample_crops(sample, settings)
                if not np.array_equal(valid_mask, sequence.valid_mask):
                    raise ValueError("motion and face-crop validity masks differ")
                source = crops[sequence.source_frame_index]
                normalized = normalizer.normalize(sequence.lip_vector)
                target_detections = tuple(active_landmarks.detect(frame) for frame in crops)
                metric_jobs: list[Future[dict[str, Any]]] = []
                examples: list[tuple[str, np.ndarray]] = []

                full_frames = active_backend.reconstruct(
                    source,
                    sequence,
                    mode="full_motion",
                )
                full_arguments = (
                    sample,
                    PerturbationCondition("full_motion", "full_motion"),
                    sequence.lip_vector,
                    sequence.lip_vector,
                    crops,
                    full_frames,
                    active_landmarks,
                    target_detections,
                )
                if metric_executor is None:
                    rows.append(_metric_row(*full_arguments))
                else:
                    metric_jobs.append(metric_executor.submit(_metric_row, *full_arguments))
                if sample_position in settings.save_sample_positions:
                    examples.append(("full_motion", full_frames))

                for condition in active_conditions:
                    perturbed_normalized = apply_perturbation(
                        normalized,
                        condition,
                        source_frame_index=sequence.source_frame_index,
                    )
                    perturbed = normalizer.denormalize(perturbed_normalized)
                    perturbed[sequence.source_frame_index] = 0
                    reconstructed = active_backend.reconstruct(
                        source,
                        sequence,
                        mode="lip_only",
                        lip_vector=perturbed,
                    )
                    metric_arguments = (
                        sample,
                        condition,
                        sequence.lip_vector,
                        perturbed,
                        crops,
                        reconstructed,
                        active_landmarks,
                        target_detections,
                    )
                    if metric_executor is None:
                        rows.append(_metric_row(*metric_arguments))
                    else:
                        metric_jobs.append(metric_executor.submit(_metric_row, *metric_arguments))
                    if sample_position in settings.save_sample_positions:
                        examples.append((condition.name, reconstructed))
                for future in metric_jobs:
                    rows.append(future.result())
                for condition_name, reconstructed in examples:
                    _save_examples(
                        run_directory,
                        sample,
                        condition_name,
                        crops,
                        reconstructed,
                        sequence.fps,
                    )
                print(
                    f"[motion-sensitivity] completed {sample.sample_id}",
                    file=sys.stderr,
                    flush=True,
                )
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                failures.append(
                    FailureRecord(
                        sample_id=sample.sample_id,
                        speaker_id=sample.speaker_id,
                        stage="motion_sensitivity",
                        reason=str(exc),
                    )
                )
    finally:
        if metric_executor is not None:
            metric_executor.shutdown(wait=True, cancel_futures=True)
        if owns_backend:
            active_backend.close()
        if owns_landmarks:
            active_landmarks.close()

    _atomic_write_jsonl(run_directory / "results.jsonl", rows)
    summary = _summarize_results(rows, failures)
    atomic_write_json(run_directory / "summary.json", summary)
    _write_summary_csv(run_directory / "summary.csv", summary)
    write_failures(run_directory / "failures.jsonl", failures)
    _write_plots(run_directory / "plots", summary)
    return run_directory, summary, failures


def _load_sample_motion(sample: GridSample, settings: MotionSettings):
    if sample.motion_path is None:
        raise ValueError("motion_path is missing; run extract_motion.py first")
    path = resolve_record_path(
        sample.motion_path,
        settings.data_settings.data_root,
    )
    return load_motion_sequence(path)


def _load_sample_crops(
    sample: GridSample,
    settings: MotionSettings,
) -> tuple[np.ndarray, np.ndarray]:
    if sample.face_crop_path is None:
        raise ValueError("face_crop_path is missing")
    path = resolve_record_path(
        sample.face_crop_path,
        settings.data_settings.data_root,
    )
    with np.load(path, allow_pickle=False) as data:
        return data["crops"].astype(np.uint8), data["valid_mask"].astype(np.bool_)


def _metric_row(
    sample: GridSample,
    condition: PerturbationCondition,
    target_motion: np.ndarray,
    candidate_motion: np.ndarray,
    target_frames: np.ndarray,
    reconstructed_frames: np.ndarray,
    landmark_backend: FaceLandmarkBackend,
    target_detections: tuple[FaceDetection | None, ...] | None = None,
) -> dict[str, Any]:
    motion = compute_motion_metrics(target_motion, candidate_motion)
    reconstruction = compute_reconstruction_metrics(
        target_frames,
        reconstructed_frames,
        landmark_backend=landmark_backend,
        target_detections=target_detections,
    )
    return {
        "sample_id": sample.sample_id,
        "speaker_id": sample.speaker_id,
        "split": sample.split,
        "condition": condition.name,
        "family": condition.family,
        "value": condition.value,
        "seed": condition.seed,
        **{f"motion_{key}": value for key, value in motion.to_dict().items()},
        **{f"reconstruction_{key}": value for key, value in reconstruction.to_dict().items()},
    }


def _new_run_directory(root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return root / timestamp


def _write_run_metadata(run_directory: Path, settings: MotionSettings) -> None:
    atomic_write_json(run_directory / "resolved_config.json", dict(settings.config))
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    atomic_write_json(
        run_directory / "environment.json",
        {
            "git_commit": commit,
            "backend": settings.backend,
            "backend_revision": settings.backend_revision,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "random_seeds": [42, 43, 44],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.replace(temporary, path)


def _summarize_results(
    rows: list[dict[str, Any]],
    failures: list[FailureRecord],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)
    condition_summaries: list[dict[str, Any]] = []
    metric_names = (
        "motion_l1",
        "motion_rmse",
        "motion_velocity_l1",
        "reconstruction_face_mae",
        "reconstruction_psnr_db",
        "reconstruction_ssim",
        "reconstruction_mouth_mae",
        "reconstruction_mouth_nme",
        "reconstruction_landmark_coverage",
    )
    for condition_name in sorted(grouped):
        condition_rows = grouped[condition_name]
        item: dict[str, Any] = {
            "condition": condition_name,
            "family": condition_rows[0]["family"],
            "parameter_name": perturbation_parameter_name(str(condition_rows[0]["family"])),
            "value": condition_rows[0]["value"],
            "seed": condition_rows[0]["seed"],
            "sample_count": len(condition_rows),
        }
        for metric_name in metric_names:
            finite_values = np.asarray(
                [
                    row[metric_name]
                    for row in condition_rows
                    if row[metric_name] is not None and np.isfinite(float(row[metric_name]))
                ],
                dtype=np.float64,
            )
            item[f"{metric_name}_mean"] = (
                float(finite_values.mean()) if finite_values.size else None
            )
            item[f"{metric_name}_std"] = float(finite_values.std()) if finite_values.size else None
        condition_summaries.append(item)
    return {
        "summary_schema_version": 2,
        "result_count": len(rows),
        "condition_count": len(grouped),
        "failure_count": len(failures),
        "conditions": condition_summaries,
    }


def _write_summary_csv(path: Path, summary: Mapping[str, Any]) -> None:
    conditions = summary.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
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
        writer = csv.DictWriter(handle, fieldnames=list(conditions[0]))
        writer.writeheader()
        writer.writerows(conditions)
    os.replace(temporary, path)


def _save_examples(
    run_directory: Path,
    sample: GridSample,
    condition_name: str,
    target: np.ndarray,
    reconstructed: np.ndarray,
    fps: float,
) -> None:
    example_root = run_directory / "examples" / sample.sample_id
    example_root.mkdir(parents=True, exist_ok=True)
    _write_rgb_video(
        example_root / f"{condition_name}.mp4",
        reconstructed,
        fps,
    )
    frame_indices = sorted({0, target.shape[0] // 2, target.shape[0] - 1})
    panels: list[Image.Image] = []
    for frame_index in frame_indices:
        panels.append(Image.fromarray(target[frame_index]))
        panels.append(Image.fromarray(reconstructed[frame_index]))
    width, height = panels[0].size
    sheet = Image.new("RGB", (width * 2, height * len(frame_indices)))
    for panel_index, panel in enumerate(panels):
        sheet.paste(
            panel,
            ((panel_index % 2) * width, (panel_index // 2) * height),
        )
    _atomic_save_image(example_root / f"{condition_name}.jpg", sheet)


def _write_rgb_video(path: Path, frames: np.ndarray, fps: float) -> None:
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError("video frames must be uint8 RGB [T, H, W, 3]")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".mp4",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s:v",
                f"{width}x{height}",
                "-r",
                f"{fps:g}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(temporary),
            ],
            input=frames.tobytes(),
            check=True,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=path.suffix,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        image.save(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_plots(plot_root: Path, summary: Mapping[str, Any]) -> None:
    conditions = summary.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for sensitivity plots; install requirements/base.txt"
        ) from exc
    plot_root.mkdir(parents=True, exist_ok=True)
    plot_definitions = {
        "gaussian": ("Noise standard deviation (sigma)", "Gaussian Noise"),
        "quantization": ("Quantization bits", "Uniform Quantization"),
        "random_dropout": ("Keep ratio", "Random Dropout"),
        "magnitude_sparsity": ("Keep ratio", "Magnitude Sparsity"),
    }
    for family, (x_label, title) in plot_definitions.items():
        family_rows = [
            row for row in conditions if row["family"] == family and row["value"] is not None
        ]
        if not family_rows:
            continue
        grouped: dict[float, list[float]] = defaultdict(list)
        for row in family_rows:
            metric = row["reconstruction_psnr_db_mean"]
            if metric is not None:
                grouped[float(row["value"])].append(float(metric))
        x_values = sorted(grouped)
        y_values = [float(np.mean(grouped[value])) for value in x_values]
        figure, axis = plt.subplots(figsize=(5, 3.5))
        axis.plot(x_values, y_values, marker="o")
        axis.set_xlabel(x_label)
        axis.set_ylabel("PSNR (dB)")
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        output = plot_root / f"{family}_psnr.png"
        with tempfile.NamedTemporaryFile(
            dir=plot_root,
            prefix=f".{family}.",
            suffix=".png",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            figure.savefig(temporary, dpi=160)
            os.replace(temporary, output)
        finally:
            plt.close(figure)
            temporary.unlink(missing_ok=True)
