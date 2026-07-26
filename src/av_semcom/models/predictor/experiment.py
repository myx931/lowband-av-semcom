"""Training and motion-level evaluation for the causal GRU baseline."""

from __future__ import annotations

import csv
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
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from av_semcom.data.grid import GridSample
from av_semcom.data.preprocessing import atomic_write_json, config_fingerprint
from av_semcom.metrics.motion import compute_motion_metrics
from av_semcom.models.motion.perturbations import (
    MotionNormalizer,
    load_motion_normalizer,
)
from av_semcom.models.predictor.artifacts import (
    atomic_save_checkpoint,
    file_sha256,
    load_checkpoint,
    save_prediction,
)
from av_semcom.models.predictor.baselines import baseline_prediction
from av_semcom.models.predictor.config import AudioMotionSettings
from av_semcom.models.predictor.data import (
    AudioMotionDataset,
    AudioNormalizer,
    audit_predictor_samples,
    fit_audio_normalizer,
    load_audio_motion_pair,
    save_audio_normalizer,
)
from av_semcom.models.predictor.model import AudioToMotionGRU, masked_l1_loss
from av_semcom.utils.reproducibility import seed_everything


def run_audio_motion_experiment(
    settings: AudioMotionSettings,
    samples: Sequence[GridSample],
    *,
    run_directory: Path | None = None,
    resume: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Train all configured seeds and evaluate learned and fixed baselines."""

    audit = audit_predictor_samples(samples, settings.data_settings.data_root)
    if audit.errors:
        raise ValueError("predictor data audit failed: " + "; ".join(audit.errors))
    expected_splits = {"train", "validation", "test"}
    if set(audit.split_counts) != expected_splits:
        raise ValueError(f"predictor data must contain splits {sorted(expected_splits)}")

    train_samples = [sample for sample in samples if sample.split == "train"]
    evaluation_samples = [
        sample for sample in samples if sample.split in settings.evaluation_splits
    ]
    motion_normalizer = load_motion_normalizer(settings.motion_stats_path)
    if motion_normalizer.scope != "train_stats":
        raise ValueError("motion normalizer must have scope train_stats")
    audio_normalizer = fit_audio_normalizer(
        train_samples,
        settings.data_settings.data_root,
    )

    run_dir = run_directory or _new_run_directory(settings.output_root)
    base_fingerprint = _base_fingerprint(settings)
    _prepare_run_directory(
        run_dir,
        settings,
        audit.to_dict(),
        audio_normalizer,
        base_fingerprint=base_fingerprint,
        resume=resume,
    )
    experiment_fingerprint = config_fingerprint(
        {
            "base_fingerprint": base_fingerprint,
            "audio_stats": audio_normalizer.to_dict(),
        }
    )
    atomic_write_json(
        run_dir / "experiment.json",
        {
            "experiment_fingerprint": experiment_fingerprint,
            "base_fingerprint": base_fingerprint,
            "status": "running",
        },
    )

    rows: list[dict[str, Any]] = []
    baseline_marker = run_dir / "baselines_complete.json"
    if resume and baseline_marker.is_file():
        _require_marker(baseline_marker, experiment_fingerprint)
        rows.extend(_read_jsonl(run_dir / "baseline_motion_metrics.jsonl"))
    else:
        baseline_rows = _evaluate_baselines(
            evaluation_samples,
            settings,
            motion_normalizer,
            run_dir,
            experiment_fingerprint,
        )
        _atomic_write_jsonl(run_dir / "baseline_motion_metrics.jsonl", baseline_rows)
        atomic_write_json(
            baseline_marker,
            {"experiment_fingerprint": experiment_fingerprint, "row_count": len(baseline_rows)},
        )
        rows.extend(baseline_rows)

    for seed in settings.seeds:
        seed_dir = run_dir / f"seed_{seed}"
        marker = seed_dir / "complete.json"
        if resume and marker.is_file():
            _require_marker(marker, experiment_fingerprint)
            rows.extend(_read_jsonl(seed_dir / "motion_metrics.jsonl"))
            continue
        seed_rows = _train_one_seed(
            seed,
            train_samples,
            evaluation_samples,
            settings,
            audio_normalizer,
            motion_normalizer,
            seed_dir,
            experiment_fingerprint,
        )
        rows.extend(seed_rows)

    summary = _summarize_motion_rows(rows, settings.seeds)
    _atomic_write_jsonl(run_dir / "motion_metrics.jsonl", rows)
    atomic_write_json(run_dir / "summary.json", summary)
    _write_summary_csv(run_dir / "summary.csv", summary)
    _write_training_plot(run_dir, settings.seeds)
    atomic_write_json(
        run_dir / "experiment.json",
        {
            "experiment_fingerprint": experiment_fingerprint,
            "base_fingerprint": base_fingerprint,
            "status": "complete",
            "result_count": len(rows),
        },
    )
    return run_dir, summary


def _train_one_seed(
    seed: int,
    train_samples: Sequence[GridSample],
    evaluation_samples: Sequence[GridSample],
    settings: AudioMotionSettings,
    audio_normalizer: AudioNormalizer,
    motion_normalizer: MotionNormalizer,
    seed_dir: Path,
    experiment_fingerprint: str,
) -> list[dict[str, Any]]:
    seed_everything(seed, deterministic=settings.deterministic)
    device = _resolve_device(settings.device)
    train_dataset = AudioMotionDataset(
        train_samples,
        settings.data_settings.data_root,
        audio_normalizer,
        motion_normalizer,
    )
    validation_samples = [sample for sample in evaluation_samples if sample.split == "validation"]
    validation_dataset = AudioMotionDataset(
        validation_samples,
        settings.data_settings.data_root,
        audio_normalizer,
        motion_normalizer,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = _build_model(settings).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    use_amp = settings.mixed_precision and device.type == "cuda"
    scaler_factory = getattr(torch.amp, "GradScaler")
    scaler: Any = scaler_factory("cuda", enabled=use_amp)
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = seed_dir / "best.pt"

    for epoch in range(1, settings.max_epochs + 1):
        train_loss = _train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            gradient_clip_norm=settings.gradient_clip_norm,
            use_amp=use_amp,
        )
        validation_loss = _evaluate_loss(model, validation_loader, device)
        if not np.isfinite(train_loss) or not np.isfinite(validation_loss):
            raise RuntimeError(f"non-finite loss at seed {seed}, epoch {epoch}")
        history.append(
            {
                "epoch": epoch,
                "train_l1": train_loss,
                "validation_l1": validation_loss,
            }
        )
        print(
            f"[audio-motion] seed={seed} epoch={epoch} "
            f"train_l1={train_loss:.6f} validation_l1={validation_loss:.6f}",
            flush=True,
        )
        if validation_loss < best_loss - settings.early_stopping_min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            atomic_save_checkpoint(
                checkpoint_path,
                {
                    "experiment_fingerprint": experiment_fingerprint,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_l1": validation_loss,
                    "model_state": model.state_dict(),
                    "model_config": _model_config(settings),
                },
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= settings.early_stopping_patience:
                break

    _atomic_write_jsonl(seed_dir / "history.jsonl", history)
    checkpoint = load_checkpoint(
        checkpoint_path,
        expected_fingerprint=experiment_fingerprint,
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state"])
    rows = _evaluate_model(
        model,
        evaluation_samples,
        settings,
        audio_normalizer,
        motion_normalizer,
        seed_dir,
        seed,
        experiment_fingerprint,
        device,
    )
    _atomic_write_jsonl(seed_dir / "motion_metrics.jsonl", rows)
    atomic_write_json(
        seed_dir / "complete.json",
        {
            "experiment_fingerprint": experiment_fingerprint,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_l1": best_loss,
            "row_count": len(rows),
        },
    )
    return rows


def _train_epoch(
    model: AudioToMotionGRU,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    *,
    gradient_clip_norm: float,
    use_amp: bool,
) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        audio = batch["audio"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            prediction = model(audio)
            loss = masked_l1_loss(prediction, target, mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        batch_size = audio.shape[0]
        total += float(loss.detach().cpu()) * batch_size
        count += batch_size
    return total / count


@torch.no_grad()
def _evaluate_loss(
    model: AudioToMotionGRU,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        audio = batch["audio"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        loss = masked_l1_loss(model(audio), target, mask)
        batch_size = audio.shape[0]
        total += float(loss.cpu()) * batch_size
        count += batch_size
    return total / count


@torch.no_grad()
def _evaluate_model(
    model: AudioToMotionGRU,
    samples: Sequence[GridSample],
    settings: AudioMotionSettings,
    audio_normalizer: AudioNormalizer,
    motion_normalizer: MotionNormalizer,
    seed_dir: Path,
    seed: int,
    experiment_fingerprint: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for sample in samples:
        audio, target, mask = load_audio_motion_pair(
            sample,
            settings.data_settings.data_root,
        )
        normalized_audio = audio_normalizer.normalize(audio)
        tensor = torch.from_numpy(normalized_audio).unsqueeze(0).to(device)
        normalized_prediction = model(tensor)[0].float().cpu().numpy()
        prediction = motion_normalizer.denormalize(normalized_prediction)
        prediction[0] = 0
        output = seed_dir / "predictions" / sample.split / f"{sample.sample_id}.npz"
        save_prediction(
            output,
            sample_id=sample.sample_id,
            method="audio_gru",
            split=sample.split,
            speaker_id=sample.speaker_id,
            prediction=prediction,
            target=target,
            valid_mask=mask,
            seed=seed,
            experiment_fingerprint=experiment_fingerprint,
        )
        rows.append(
            _motion_metric_row(
                sample,
                method="audio_gru",
                seed=seed,
                target=target,
                prediction=prediction,
                mask=mask,
            )
        )
    return rows


def _evaluate_baselines(
    samples: Sequence[GridSample],
    settings: AudioMotionSettings,
    motion_normalizer: MotionNormalizer,
    run_dir: Path,
    experiment_fingerprint: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        _, target, mask = load_audio_motion_pair(sample, settings.data_settings.data_root)
        for method in settings.baselines:
            prediction = baseline_prediction(method, target, motion_normalizer.mean)
            output = run_dir / "predictions" / method / sample.split / f"{sample.sample_id}.npz"
            save_prediction(
                output,
                sample_id=sample.sample_id,
                method=method,
                split=sample.split,
                speaker_id=sample.speaker_id,
                prediction=prediction,
                target=target,
                valid_mask=mask,
                seed=None,
                experiment_fingerprint=experiment_fingerprint,
            )
            rows.append(
                _motion_metric_row(
                    sample,
                    method=method,
                    seed=None,
                    target=target,
                    prediction=prediction,
                    mask=mask,
                )
            )
    return rows


def _motion_metric_row(
    sample: GridSample,
    *,
    method: str,
    seed: int | None,
    target: NDArray[np.float32],
    prediction: NDArray[np.float32],
    mask: NDArray[np.bool_],
) -> dict[str, Any]:
    metrics = compute_motion_metrics(target[mask], prediction[mask])
    return {
        "sample_id": sample.sample_id,
        "speaker_id": sample.speaker_id,
        "split": sample.split,
        "method": method,
        "seed": seed,
        **metrics.to_dict(),
    }


def _summarize_motion_rows(
    rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int | None], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["split"]), row.get("seed"))].append(row)
    groups: list[dict[str, Any]] = []
    for (method, split, seed), members in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1], -1 if item[0][2] is None else item[0][2]),
    ):
        groups.append(
            {
                "method": method,
                "split": split,
                "seed": seed,
                "sample_count": len(members),
                "l1": float(np.mean([float(row["l1"]) for row in members])),
                "rmse": float(np.mean([float(row["rmse"]) for row in members])),
                "velocity_l1": float(np.mean([float(row["velocity_l1"]) for row in members])),
            }
        )
    seed_aggregate: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        seed_groups = [
            group for group in groups if group["method"] == "audio_gru" and group["split"] == split
        ]
        if len(seed_groups) != len(seeds):
            raise ValueError(f"missing audio_gru seed results for {split}")
        seed_aggregate.append(
            {
                "method": "audio_gru",
                "split": split,
                "seed_count": len(seed_groups),
                **{
                    f"{metric}_{suffix}": float(
                        function([float(group[metric]) for group in seed_groups])
                    )
                    for metric in ("l1", "rmse", "velocity_l1")
                    for suffix, function in (("mean", np.mean), ("std", np.std))
                },
            }
        )
    validation_gru = next(group for group in seed_aggregate if group["split"] == "validation")
    validation_mean = next(
        group
        for group in groups
        if group["method"] == "train_mean"
        and group["split"] == "validation"
        and group["seed"] is None
    )
    test_gru = next(group for group in seed_aggregate if group["split"] == "test")
    test_mean = next(
        group
        for group in groups
        if group["method"] == "train_mean" and group["split"] == "test" and group["seed"] is None
    )
    return {
        "schema_version": 1,
        "result_count": len(rows),
        "groups": groups,
        "seed_aggregate": seed_aggregate,
        "e4_validation_gate_passed": validation_gru["l1_mean"] < validation_mean["l1"],
        "rq1_test_mean_baseline_improved": test_gru["l1_mean"] < test_mean["l1"],
    }


def _build_model(settings: AudioMotionSettings) -> AudioToMotionGRU:
    return AudioToMotionGRU(**_model_config(settings))


def _model_config(settings: AudioMotionSettings) -> dict[str, Any]:
    return {
        "mel_bins": settings.mel_bins,
        "mel_steps_per_frame": settings.mel_steps_per_frame,
        "audio_projection_dim": settings.audio_projection_dim,
        "hidden_dim": settings.hidden_dim,
        "num_layers": settings.num_layers,
        "dropout": settings.dropout,
        "output_dim": settings.output_dim,
    }


def _resolve_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but torch.cuda.is_available() is false")
    return resolved


def _base_fingerprint(settings: AudioMotionSettings) -> str:
    return config_fingerprint(
        {
            "config": settings.config,
            "manifest_sha256": file_sha256(settings.data_settings.manifest_path),
            "motion_stats_sha256": file_sha256(settings.motion_stats_path),
        }
    )


def _prepare_run_directory(
    run_dir: Path,
    settings: AudioMotionSettings,
    audit: Mapping[str, Any],
    audio_normalizer: AudioNormalizer,
    *,
    base_fingerprint: str,
    resume: bool,
) -> None:
    metadata_path = run_dir / "run_metadata.json"
    if run_dir.exists():
        if not resume:
            raise FileExistsError(f"run directory already exists: {run_dir}")
        if not metadata_path.is_file():
            raise ValueError("run directory is missing run_metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("base_fingerprint") != base_fingerprint:
            raise ValueError("run directory was created with another configuration or data")
        return
    if resume:
        raise FileNotFoundError(f"cannot resume missing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    atomic_write_json(run_dir / "resolved_config.json", dict(settings.config))
    atomic_write_json(run_dir / "data_audit.json", dict(audit))
    save_audio_normalizer(run_dir / "audio_stats.json", audio_normalizer)
    atomic_write_json(
        metadata_path,
        {
            "base_fingerprint": base_fingerprint,
            "git_commit": _git_commit(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    atomic_write_json(run_dir / "environment.json", _environment())


def _require_marker(path: Path, fingerprint: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment_fingerprint") != fingerprint:
        raise ValueError(f"completion marker fingerprint mismatch: {path}")


def _new_run_directory(root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return root / timestamp


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
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "device_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
    }


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


def _write_summary_csv(path: Path, summary: Mapping[str, Any]) -> None:
    groups = summary["groups"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "split",
                "seed",
                "sample_count",
                "l1",
                "rmse",
                "velocity_l1",
            ],
        )
        writer.writeheader()
        writer.writerows(groups)


def _write_training_plot(run_dir: Path, seeds: Sequence[int]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axis = plt.subplots(figsize=(8, 5))
    for seed in seeds:
        rows = _read_jsonl(run_dir / f"seed_{seed}" / "history.jsonl")
        axis.plot(
            [row["epoch"] for row in rows],
            [row["validation_l1"] for row in rows],
            label=f"seed {seed}",
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Normalized validation L1")
    axis.set_title("Causal GRU validation curves")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(run_dir / "training_curves.png", dpi=160)
    plt.close(figure)
