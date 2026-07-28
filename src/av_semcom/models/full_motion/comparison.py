"""Matched-symbol comparison of frozen residual and full-motion JSCC."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from av_semcom.data.preprocessing import atomic_write_json, config_fingerprint
from av_semcom.models.jscc.export import select_validation_model_seeds
from av_semcom.models.predictor.artifacts import file_sha256
from av_semcom.models.selection.gate import (
    _environment,
    _git_commit,
    _new_run_directory,
    _read_json,
    _write_dict_csv,
)

_MOTION_METRICS = ("l1", "rmse", "velocity_l1")
_VIDEO_METRICS = (
    "oracle_mouth_mae",
    "oracle_mouth_nme",
    "oracle_psnr_db",
    "oracle_ssim",
    "oracle_landmark_coverage",
)


def run_matched_comparison(
    output_root: Path,
    residual_run_dir: Path,
    full_motion_run_dir: Path,
    *,
    frame_count: int = 75,
    reference_frame_count: int = 1,
    frame_rate: int = 25,
    run_directory: Path | None = None,
    resume: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Compare immutable rows without training, refitting, or test selection."""

    if frame_count <= reference_frame_count or frame_rate <= 0:
        raise ValueError("invalid communication accounting dimensions")
    residual_run_dir = residual_run_dir.resolve()
    full_motion_run_dir = full_motion_run_dir.resolve()
    source = _source_provenance(residual_run_dir, full_motion_run_dir)
    accounting = {
        "frame_count": frame_count,
        "reference_frame_count": reference_frame_count,
        "eligible_frame_count": frame_count - reference_frame_count,
        "frame_rate": frame_rate,
        "clip_duration_seconds": frame_count / frame_rate,
        "rate_unit": "complex_channel_symbols",
        "digital_bitrate_defined": False,
    }
    fingerprint = config_fingerprint({"source": source, "accounting": accounting})
    run_dir = (
        run_directory.resolve()
        if run_directory is not None
        else _new_run_directory(output_root.resolve())
    )
    complete_path = run_dir / "complete.json"
    if run_dir.exists():
        if not resume:
            raise FileExistsError(f"comparison exists: {run_dir}")
        complete = _read_json(complete_path)
        if complete.get("comparison_fingerprint") != fingerprint:
            raise ValueError("comparison fingerprint mismatch")
        return run_dir, _read_json(run_dir / "summary.json")
    if resume:
        raise FileNotFoundError(f"cannot resume missing comparison: {run_dir}")
    run_dir.mkdir(parents=True)

    residual_training = _read_json(residual_run_dir / "training_summary.json")
    full_training = _read_json(full_motion_run_dir / "training_summary.json")
    channel_uses = _common_channel_uses(residual_training, full_training)
    residual_selected = select_validation_model_seeds(
        residual_training,
        channel_uses,
    )
    full_selected = select_validation_model_seeds(
        full_training,
        channel_uses,
    )
    motion_rows = _motion_comparison_rows(
        _read_jsonl(residual_run_dir / "test_metrics.jsonl"),
        _read_jsonl(full_motion_run_dir / "test_metrics.jsonl"),
        residual_selected,
        full_selected,
        accounting,
    )
    motion_groups = _summarize_motion(motion_rows)
    video_rows = _video_comparison_rows(
        _read_json(residual_run_dir / "video_reconstruction/summary.json"),
        _read_json(full_motion_run_dir / "video_reconstruction/summary.json"),
        accounting,
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "comparison_fingerprint": fingerprint,
        "scope": "frozen_test_matched_complex_symbol_budget",
        "model_selection_performed": False,
        "test_metrics_recomputed": False,
        "digital_bitrate_defined": False,
        "motion_pair_count": len(motion_rows),
        "motion_group_count": len(motion_groups),
        "video_group_count": len(video_rows),
        "residual_lower_l1_group_count": sum(
            float(row["residual_advantage_l1"]) > 0 for row in motion_groups
        ),
        "full_motion_lower_l1_group_count": sum(
            float(row["residual_advantage_l1"]) < 0 for row in motion_groups
        ),
        "tied_l1_group_count": sum(
            float(row["residual_advantage_l1"]) == 0 for row in motion_groups
        ),
        "residual_lower_mouth_nme_group_count": sum(
            float(row["residual_advantage_oracle_mouth_nme"]) > 0 for row in video_rows
        ),
        "source_provenance": source,
        "accounting": accounting,
        "interpretation": (
            "positive residual_advantage means residual JSCC has lower error than full-motion JSCC"
        ),
        "noise_pairing": (
            "nominal noise seeds are matched; exact derived realization also "
            "depends on each validation-selected model seed"
        ),
    }
    atomic_write_json(run_dir / "source_provenance.json", source)
    atomic_write_json(run_dir / "accounting.json", accounting)
    atomic_write_json(run_dir / "environment.json", _environment())
    atomic_write_json(
        run_dir / "run_metadata.json",
        {
            "comparison_fingerprint": fingerprint,
            "git_commit": _git_commit(),
            "model_selection_performed": False,
            "test_metrics_recomputed": False,
        },
    )
    _atomic_write_jsonl(run_dir / "motion_pairs.jsonl", motion_rows)
    atomic_write_json(run_dir / "motion_summary.json", {"rows": motion_groups})
    _write_dict_csv(run_dir / "motion_summary.csv", motion_groups)
    atomic_write_json(run_dir / "video_summary.json", {"rows": video_rows})
    _write_dict_csv(run_dir / "video_summary.csv", video_rows)
    atomic_write_json(run_dir / "summary.json", summary)
    _write_plots(run_dir / "plots", motion_groups, video_rows)
    atomic_write_json(
        complete_path,
        {
            "status": "complete",
            "comparison_fingerprint": fingerprint,
            "motion_pair_count": len(motion_rows),
            "video_group_count": len(video_rows),
        },
    )
    return run_dir, summary


def _motion_comparison_rows(
    residual_rows: Sequence[Mapping[str, Any]],
    full_rows: Sequence[Mapping[str, Any]],
    residual_selected: Mapping[int, int],
    full_selected: Mapping[int, int],
    accounting: Mapping[str, Any],
) -> list[dict[str, Any]]:
    residual_index = _selected_motion_index(residual_rows, residual_selected)
    full_index = _selected_motion_index(full_rows, full_selected)
    if set(residual_index) != set(full_index):
        missing_residual = len(set(full_index) - set(residual_index))
        missing_full = len(set(residual_index) - set(full_index))
        raise ValueError(
            f"matched motion identities differ: residual missing {missing_residual}, "
            f"full motion missing {missing_full}"
        )
    eligible = int(accounting["eligible_frame_count"])
    duration = float(accounting["clip_duration_seconds"])
    output: list[dict[str, Any]] = []
    for key in sorted(residual_index):
        sample_id, channel_uses, snr_db, noise_seed = key
        residual = residual_index[key]
        full = full_index[key]
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "speaker_id": residual["speaker_id"],
            "split": residual["split"],
            "channel_uses": channel_uses,
            "snr_db": snr_db,
            "noise_seed": noise_seed,
            "residual_model_seed": int(residual["model_seed"]),
            "full_motion_model_seed": int(full["model_seed"]),
            "exact_derived_noise_realization_matched": (
                int(residual["model_seed"]) == int(full["model_seed"])
            ),
            "complex_symbols_per_clip": channel_uses * eligible,
            "complex_symbols_per_second": channel_uses * eligible / duration,
        }
        for metric in _MOTION_METRICS:
            residual_value = float(residual[metric])
            full_value = float(full[metric])
            row[f"residual_{metric}"] = residual_value
            row[f"full_motion_{metric}"] = full_value
            row[f"residual_advantage_{metric}"] = full_value - residual_value
        output.append(row)
    return output


def _selected_motion_index(
    rows: Sequence[Mapping[str, Any]],
    selected: Mapping[int, int],
) -> dict[tuple[str, int, float, int], Mapping[str, Any]]:
    output: dict[tuple[str, int, float, int], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("condition") != "jscc_awgn":
            continue
        channel_uses = int(row["channel_uses"])
        if int(row["model_seed"]) != selected[channel_uses]:
            continue
        key = (
            str(row["sample_id"]),
            channel_uses,
            float(row["snr_db"]),
            int(row["noise_seed"]),
        )
        if key in output:
            raise ValueError(f"duplicate matched motion row: {key}")
        output[key] = row
    if not output:
        raise ValueError("no validation-selected test motion rows")
    return output


def _summarize_motion(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["channel_uses"]), float(row["snr_db"]))].append(row)
    output: list[dict[str, Any]] = []
    for (channel_uses, snr_db), members in sorted(grouped.items()):
        sample_advantages: dict[str, list[float]] = defaultdict(list)
        for row in members:
            sample_advantages[str(row["sample_id"])].append(float(row["residual_advantage_l1"]))
        sample_means = np.asarray(
            [np.mean(values) for values in sample_advantages.values()],
            dtype=np.float64,
        )
        record: dict[str, Any] = {
            "channel_uses": channel_uses,
            "snr_db": snr_db,
            "sample_count": len(sample_advantages),
            "sample_noise_realization_count": len(members),
            "complex_symbols_per_clip": int(members[0]["complex_symbols_per_clip"]),
            "complex_symbols_per_second": float(members[0]["complex_symbols_per_second"]),
            "residual_sample_win_fraction_l1": float((sample_means > 0).mean()),
            "residual_advantage_l1_sample_std": float(sample_means.std()),
        }
        for metric in _MOTION_METRICS:
            for prefix in ("residual", "full_motion", "residual_advantage"):
                record[f"{prefix}_{metric}"] = float(
                    np.mean([float(row[f"{prefix}_{metric}"]) for row in members])
                )
        output.append(record)
    return output


def _video_comparison_rows(
    residual_summary: Mapping[str, Any],
    full_summary: Mapping[str, Any],
    accounting: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if (
        int(residual_summary.get("failure_count", -1)) != 0
        or int(full_summary.get("failure_count", -1)) != 0
    ):
        raise ValueError("video comparison source contains failures")
    residual_index = _video_index(residual_summary)
    full_index = _video_index(full_summary)
    if set(residual_index) != set(full_index):
        raise ValueError("video comparison conditions differ")
    eligible = int(accounting["eligible_frame_count"])
    duration = float(accounting["clip_duration_seconds"])
    rows: list[dict[str, Any]] = []
    for channel_uses, snr_db in sorted(residual_index):
        residual = residual_index[(channel_uses, snr_db)]
        full = full_index[(channel_uses, snr_db)]
        row: dict[str, Any] = {
            "channel_uses": channel_uses,
            "snr_db": snr_db,
            "sample_count": int(residual["sample_count"]),
            "noise_seed": int(residual["noise_seed"]),
            "complex_symbols_per_clip": channel_uses * eligible,
            "complex_symbols_per_second": channel_uses * eligible / duration,
        }
        for metric in _VIDEO_METRICS:
            residual_value = float(residual[metric])
            full_value = float(full[metric])
            row[f"residual_{metric}"] = residual_value
            row[f"full_motion_{metric}"] = full_value
            if metric in {
                "oracle_mouth_mae",
                "oracle_mouth_nme",
            }:
                row[f"residual_advantage_{metric}"] = full_value - residual_value
        rows.append(row)
    return rows


def _video_index(
    summary: Mapping[str, Any],
) -> dict[tuple[int, float], Mapping[str, Any]]:
    groups = summary.get("groups")
    if not isinstance(groups, list):
        raise ValueError("video summary has no groups")
    output = {
        (int(row["channel_uses"]), float(row["snr_db"])): row
        for row in groups
        if row.get("family") == "jscc_awgn"
    }
    if not output:
        raise ValueError("video summary has no JSCC AWGN rows")
    return output


def _source_provenance(
    residual: Path,
    full: Path,
) -> dict[str, Any]:
    for root in (residual, full):
        marker = _read_json(root / "evaluation_complete.json")
        if marker.get("status") != "complete":
            raise ValueError(f"incomplete motion evaluation: {root}")
        video = _read_json(root / "video_reconstruction/complete.json")
        if video.get("status") != "complete" or int(video.get("failure_count", -1)) != 0:
            raise ValueError(f"incomplete video evaluation: {root}")
    return {
        "residual_test_metrics_sha256": file_sha256(residual / "test_metrics.jsonl"),
        "residual_training_summary_sha256": file_sha256(residual / "training_summary.json"),
        "residual_video_summary_sha256": file_sha256(
            residual / "video_reconstruction/summary.json"
        ),
        "full_motion_test_metrics_sha256": file_sha256(full / "test_metrics.jsonl"),
        "full_motion_training_summary_sha256": file_sha256(full / "training_summary.json"),
        "full_motion_video_summary_sha256": file_sha256(full / "video_reconstruction/summary.json"),
    }


def _common_channel_uses(
    residual: Mapping[str, Any],
    full: Mapping[str, Any],
) -> tuple[int, ...]:
    def values(summary: Mapping[str, Any]) -> set[int]:
        models = summary.get("models")
        if not isinstance(models, list):
            raise ValueError("training summary has no models")
        return {int(row["channel_uses"]) for row in models}

    residual_values = values(residual)
    full_values = values(full)
    if not residual_values or residual_values != full_values:
        raise ValueError("residual and full-motion channel budgets differ")
    return tuple(sorted(residual_values))


def _write_plots(
    path: Path,
    motion: Sequence[Mapping[str, Any]],
    video: Sequence[Mapping[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    path.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for snr_db in sorted({float(row["snr_db"]) for row in motion}):
        members = [row for row in motion if float(row["snr_db"]) == snr_db]
        axes[0].plot(
            [row["complex_symbols_per_second"] for row in members],
            [row["residual_l1"] for row in members],
            marker="o",
            label=f"residual {snr_db:g} dB",
        )
        axes[0].plot(
            [row["complex_symbols_per_second"] for row in members],
            [row["full_motion_l1"] for row in members],
            marker="x",
            linestyle="--",
            label=f"full {snr_db:g} dB",
        )
    for snr_db in sorted({float(row["snr_db"]) for row in video}):
        members = [row for row in video if float(row["snr_db"]) == snr_db]
        axes[1].plot(
            [row["complex_symbols_per_second"] for row in members],
            [row["residual_oracle_mouth_nme"] for row in members],
            marker="o",
            label=f"residual {snr_db:g} dB",
        )
        axes[1].plot(
            [row["complex_symbols_per_second"] for row in members],
            [row["full_motion_oracle_mouth_nme"] for row in members],
            marker="x",
            linestyle="--",
            label=f"full {snr_db:g} dB",
        )
    axes[0].set_ylabel("raw motion L1")
    axes[1].set_ylabel("mouth NME vs oracle")
    for axis in axes:
        axis.set_xlabel("complex channel symbols / second")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    figure.savefig(path / "full_motion_vs_residual.png", dpi=160)
    plt.close(figure)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    from av_semcom.models.jscc.experiment import _atomic_write_jsonl as write

    write(path, rows)
