"""Full validation/test reconstruction evaluation for E3 predictions."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from av_semcom.data.grid import GridSample, resolve_record_path
from av_semcom.data.landmarks import FaceLandmarkBackend, MediaPipeFaceMeshBackend
from av_semcom.data.preprocessing import FailureRecord, atomic_write_json, write_failures
from av_semcom.metrics.motion import compute_reconstruction_metrics
from av_semcom.models.motion.config import MotionSettings
from av_semcom.models.motion.pipeline import create_reconstruction_backend
from av_semcom.models.motion.sequence import load_motion_sequence
from av_semcom.models.predictor.artifacts import load_prediction
from av_semcom.models.predictor.config import AudioMotionSettings
from av_semcom.models.reconstruction.backend import ReconstructionBackend


def run_prediction_reconstruction(
    settings: AudioMotionSettings,
    motion_settings: MotionSettings,
    samples: Sequence[GridSample],
    run_dir: Path,
    *,
    resume: bool = False,
    backend: ReconstructionBackend | None = None,
    landmark_backend: FaceLandmarkBackend | None = None,
    save_representative_media: bool = True,
) -> tuple[dict[str, Any], list[FailureRecord]]:
    """Render and score all deterministic baselines and GRU seeds."""

    experiment = json.loads((run_dir / "experiment.json").read_text(encoding="utf-8"))
    if experiment.get("status") != "complete":
        raise ValueError("audio-to-motion run is not complete")
    fingerprint = str(experiment.get("experiment_fingerprint", ""))
    if not fingerprint:
        raise ValueError("audio-to-motion run has no experiment fingerprint")
    evaluation_samples = sorted(
        (sample for sample in samples if sample.split in settings.evaluation_splits),
        key=lambda sample: (sample.split, sample.sample_id),
    )
    if not evaluation_samples:
        raise ValueError("reconstruction evaluation has no validation/test samples")
    summary_path = run_dir / "reconstruction" / "summary.json"
    if resume and (run_dir / "reconstruction" / "complete.json").is_file():
        marker = json.loads(
            (run_dir / "reconstruction" / "complete.json").read_text(encoding="utf-8")
        )
        if marker.get("experiment_fingerprint") != fingerprint:
            raise ValueError("reconstruction completion fingerprint does not match")
        return json.loads(summary_path.read_text(encoding="utf-8")), []

    best_seed = _best_validation_seed(run_dir)
    representative_ids = _representative_sample_ids(evaluation_samples)
    output_root = run_dir / "reconstruction"
    rows: list[dict[str, Any]] = []
    failures: list[FailureRecord] = []
    owns_backend = backend is None
    active_backend = backend or create_reconstruction_backend(motion_settings)
    owns_landmarks = landmark_backend is None
    landmarks = landmark_backend or MediaPipeFaceMeshBackend()
    try:
        for position, sample in enumerate(evaluation_samples, start=1):
            print(
                f"[audio-motion-reconstruction] sample {position}/{len(evaluation_samples)}: "
                f"{sample.sample_id}",
                flush=True,
            )
            sample_result = output_root / "samples" / f"{sample.sample_id}.json"
            if sample_result.exists():
                if not resume:
                    raise FileExistsError(
                        f"reconstruction sample result already exists: {sample_result}"
                    )
                payload = json.loads(sample_result.read_text(encoding="utf-8"))
                if payload.get("experiment_fingerprint") != fingerprint:
                    raise ValueError(f"stale reconstruction sample result: {sample_result}")
                rows.extend(payload["rows"])
                continue
            try:
                sample_rows = _evaluate_reconstruction_sample(
                    sample,
                    settings,
                    active_backend,
                    landmarks,
                    run_dir,
                    fingerprint,
                    best_seed=best_seed,
                    save_media=(
                        save_representative_media and sample.sample_id in representative_ids
                    ),
                    output_root=output_root,
                )
                atomic_write_json(
                    sample_result,
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
                        stage="audio_motion_reconstruction",
                        reason=str(exc),
                    )
                )
    finally:
        if owns_backend:
            active_backend.close()
        if owns_landmarks:
            landmarks.close()

    _atomic_write_jsonl(output_root / "metrics.jsonl", rows)
    write_failures(output_root / "failures.jsonl", failures)
    summary = _summarize(rows, failures)
    atomic_write_json(summary_path, summary)
    _write_summary_csv(output_root / "summary.csv", summary)
    _write_plot(output_root / "metric_comparison.png", summary)
    if not failures:
        atomic_write_json(
            output_root / "complete.json",
            {
                "experiment_fingerprint": fingerprint,
                "sample_count": len(evaluation_samples),
                "result_count": len(rows),
                "best_validation_seed": best_seed,
            },
        )
    return summary, failures


def _evaluate_reconstruction_sample(
    sample: GridSample,
    settings: AudioMotionSettings,
    backend: ReconstructionBackend,
    landmarks: FaceLandmarkBackend,
    run_dir: Path,
    fingerprint: str,
    *,
    best_seed: int,
    save_media: bool,
    output_root: Path,
) -> list[dict[str, Any]]:
    if sample.motion_path is None or sample.face_crop_path is None:
        raise ValueError("motion_path and face_crop_path are required")
    sequence = load_motion_sequence(
        resolve_record_path(sample.motion_path, settings.data_settings.data_root)
    )
    with np.load(
        resolve_record_path(sample.face_crop_path, settings.data_settings.data_root),
        allow_pickle=False,
    ) as payload:
        original = payload["crops"].astype(np.uint8)
    source = original[sequence.source_frame_index]
    oracle = backend.reconstruct(
        source,
        sequence,
        mode="lip_only",
        lip_vector=sequence.lip_vector,
    )
    original_detections = tuple(landmarks.detect(frame) for frame in original)
    oracle_detections = tuple(landmarks.detect(frame) for frame in oracle)
    if save_media:
        media_root = output_root / "media" / sample.split / sample.sample_id
        _save_video(media_root / "original.mp4", original, sequence.fps)
        _save_video(media_root / "oracle_lip.mp4", oracle, sequence.fps)

    rows: list[dict[str, Any]] = []
    methods: list[tuple[str, int | None, Path]] = []
    for method in settings.baselines:
        methods.append(
            (
                method,
                None,
                run_dir / "predictions" / method / sample.split / f"{sample.sample_id}.npz",
            )
        )
    for seed in settings.seeds:
        methods.append(
            (
                "audio_gru",
                seed,
                run_dir / f"seed_{seed}" / "predictions" / sample.split / f"{sample.sample_id}.npz",
            )
        )
    for method, method_seed, prediction_path in methods:
        prediction = load_prediction(
            prediction_path,
            expected_fingerprint=fingerprint,
        )["prediction"]
        reconstructed = backend.reconstruct(
            source,
            sequence,
            mode="lip_only",
            lip_vector=prediction,
        )
        reconstructed_detections = tuple(landmarks.detect(frame) for frame in reconstructed)
        oracle_metrics = compute_reconstruction_metrics(
            oracle,
            reconstructed,
            landmark_backend=landmarks,
            target_detections=oracle_detections,
            reconstructed_detections=reconstructed_detections,
        )
        original_metrics = compute_reconstruction_metrics(
            original,
            reconstructed,
            landmark_backend=landmarks,
            target_detections=original_detections,
            reconstructed_detections=reconstructed_detections,
        )
        row = {
            "sample_id": sample.sample_id,
            "speaker_id": sample.speaker_id,
            "split": sample.split,
            "method": method,
            "seed": method_seed,
            **{f"oracle_{key}": value for key, value in oracle_metrics.to_dict().items()},
            **{f"original_{key}": value for key, value in original_metrics.to_dict().items()},
        }
        rows.append(row)
        if save_media and (method != "audio_gru" or method_seed == best_seed):
            label = method if method_seed is None else f"audio_gru_seed_{method_seed}"
            media_root = output_root / "media" / sample.split / sample.sample_id
            _save_video(media_root / f"{label}.mp4", reconstructed, sequence.fps)
            _save_contact_sheet(
                media_root / f"{label}.jpg",
                original,
                oracle,
                reconstructed,
            )
    return rows


def _representative_sample_ids(samples: Sequence[GridSample]) -> set[str]:
    selected: set[str] = set()
    for split in ("validation", "test"):
        split_samples = [sample for sample in samples if sample.split == split]
        if split_samples:
            for index in (0, len(split_samples) // 2, len(split_samples) - 1):
                selected.add(split_samples[index].sample_id)
    return selected


def _best_validation_seed(run_dir: Path) -> int:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    candidates = [
        group
        for group in summary["groups"]
        if group["method"] == "audio_gru" and group["split"] == "validation"
    ]
    if not candidates:
        raise ValueError("run summary has no validation GRU results")
    return int(min(candidates, key=lambda group: float(group["l1"]))["seed"])


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    failures: Sequence[FailureRecord],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int | None], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["split"]), row.get("seed"))].append(row)
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
    for (method, split, seed), members in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1], -1 if item[0][2] is None else item[0][2]),
    ):
        group: dict[str, Any] = {
            "method": method,
            "split": split,
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


def _save_video(path: Path, frames: NDArray[np.uint8], fps: float) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    imageio.mimsave(
        path,
        [frame for frame in frames],
        fps=fps,
        codec="libx264",
        quality=8,
    )


def _save_contact_sheet(
    path: Path,
    original: NDArray[np.uint8],
    oracle: NDArray[np.uint8],
    reconstructed: NDArray[np.uint8],
) -> None:
    if path.exists():
        return
    indices = (0, original.shape[0] // 2, original.shape[0] - 1)
    canvas = Image.new("RGB", (3 * 256, 3 * 256))
    for row, frames in enumerate((original, oracle, reconstructed)):
        for column, frame_index in enumerate(indices):
            canvas.paste(Image.fromarray(frames[frame_index]), (column * 256, row * 256))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


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


def _write_plot(path: Path, summary: Mapping[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    groups = [
        group
        for group in summary["groups"]
        if group["split"] == "test" and group["seed"] in {None, 42, 43, 44}
    ]
    if not groups:
        return
    labels = [
        group["method"] if group["seed"] is None else f"GRU {group['seed']}" for group in groups
    ]
    values = [group["oracle_mouth_nme"] for group in groups]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(labels, values)
    axis.set_ylabel("Oracle-relative mouth NME")
    axis.set_title("E3 test reconstruction comparison")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
