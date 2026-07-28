"""Build a read-only thesis evidence pack from frozen E3-E7 artifacts."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from av_semcom.data.preprocessing import atomic_write_json, config_fingerprint
from av_semcom.models.predictor.artifacts import file_sha256
from av_semcom.models.selection.gate import (
    _environment,
    _git_commit,
    _new_run_directory,
    _read_json,
    _read_jsonl,
    _write_dict_csv,
)
from av_semcom.utils.config import ConfigError

_MOTION_METRICS = ("l1", "rmse", "velocity_l1")
_VIDEO_METRICS = ("oracle_mouth_mae", "oracle_mouth_nme")
_CHANNEL_USES = (1, 2, 3, 4)
_TEST_SNR_DB = (-5.0, 0.0, 5.0, 10.0)


@dataclass(frozen=True)
class ThesisEvidenceSettings:
    """Frozen reporting and paired-bootstrap settings."""

    output_root: Path
    bootstrap_seed: int
    bootstrap_resamples: int
    confidence_level: float
    expected_test_sample_count: int
    qualitative_positions: tuple[int, ...]
    figure_dpi: int
    config: Mapping[str, Any]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ThesisEvidenceSettings:
        raw = config.get("thesis_evidence")
        if not isinstance(raw, Mapping):
            raise ConfigError("thesis_evidence configuration must be a mapping")
        output_raw = raw.get("output_dir", "outputs/thesis_evidence")
        if not isinstance(output_raw, str) or not output_raw:
            raise ConfigError("thesis_evidence.output_dir must be a path")
        output_root = Path(output_raw)
        if not output_root.is_absolute():
            output_root = Path(__file__).resolve().parents[3] / output_root
        bootstrap_seed = int(raw.get("bootstrap_seed", 42))
        bootstrap_resamples = int(raw.get("bootstrap_resamples", 0))
        confidence_level = float(raw.get("confidence_level", 0.0))
        expected_count = int(raw.get("expected_test_sample_count", 0))
        positions_raw = raw.get("qualitative_positions")
        figure_dpi = int(raw.get("figure_dpi", 0))
        if bootstrap_resamples < 100:
            raise ConfigError("thesis_evidence.bootstrap_resamples must be at least 100")
        if not 0.5 < confidence_level < 1.0:
            raise ConfigError("thesis_evidence.confidence_level must be between 0.5 and 1")
        if expected_count <= 1:
            raise ConfigError("thesis_evidence.expected_test_sample_count must exceed 1")
        if not isinstance(positions_raw, list) or not positions_raw:
            raise ConfigError("thesis_evidence.qualitative_positions must be a non-empty list")
        positions = tuple(int(value) for value in positions_raw)
        if len(set(positions)) != len(positions) or min(positions) < 0:
            raise ConfigError("qualitative positions must be unique and non-negative")
        if max(positions) >= expected_count:
            raise ConfigError("qualitative position exceeds expected test sample count")
        if figure_dpi < 72:
            raise ConfigError("thesis_evidence.figure_dpi must be at least 72")
        return cls(
            output_root=output_root.resolve(),
            bootstrap_seed=bootstrap_seed,
            bootstrap_resamples=bootstrap_resamples,
            confidence_level=confidence_level,
            expected_test_sample_count=expected_count,
            qualitative_positions=positions,
            figure_dpi=figure_dpi,
            config=dict(raw),
        )


@dataclass(frozen=True)
class ThesisSourceRuns:
    """Locations of immutable formal runs consumed by E8."""

    e3: Path
    e4: Path
    residual_jscc: Path
    gate: Path
    scorer: Path
    scorer_ablation: Path
    communication: Path
    full_motion: Path
    comparison: Path

    def resolved(self) -> ThesisSourceRuns:
        return ThesisSourceRuns(
            **{name: Path(value).resolve() for name, value in self.__dict__.items()}
        )


def paired_bootstrap_mean(
    values: Sequence[float] | np.ndarray,
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Return a deterministic percentile interval for one paired-difference vector."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError("paired bootstrap requires at least two finite one-dimensional values")
    if resamples < 100 or not 0.5 < confidence_level < 1.0:
        raise ValueError("invalid paired bootstrap configuration")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    bootstrap_means = array[indices].mean(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(bootstrap_means, [tail, 1.0 - tail])
    point = float(array.mean())
    return {
        "sample_count": int(array.size),
        "point_estimate": point,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence_level": confidence_level,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "sample_win_fraction": float((array > 0).mean()),
        "ci_excludes_zero": bool(lower > 0 or upper < 0),
        "direction": (
            "residual_better"
            if lower > 0
            else "full_motion_better"
            if upper < 0
            else "inconclusive"
        ),
    }


def summarize_motion_pairs(
    rows: Sequence[Mapping[str, Any]],
    settings: ThesisEvidenceSettings,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Average noise within sample, then bootstrap across independent samples."""

    grouped: dict[tuple[int, float, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                int(row["channel_uses"]),
                float(row["snr_db"]),
                str(row["sample_id"]),
            )
        ].append(row)
    conditions = {(key[0], key[1]) for key in grouped}
    if conditions != {(c, snr) for c in _CHANNEL_USES for snr in _TEST_SNR_DB}:
        raise ValueError("E7 motion pairs do not contain the frozen C x SNR grid")
    sample_rows: list[dict[str, Any]] = []
    expected_noise_count: int | None = None
    for (channel_uses, snr_db, sample_id), members in sorted(grouped.items()):
        noise_count = len(members)
        if expected_noise_count is None:
            expected_noise_count = noise_count
        noise_seeds = {int(row["noise_seed"]) for row in members}
        if noise_count != expected_noise_count or len(noise_seeds) != noise_count:
            raise ValueError("E7 motion noise realizations are incomplete or duplicated")
        sample_rows.append(
            {
                "sample_id": sample_id,
                "channel_uses": channel_uses,
                "snr_db": snr_db,
                "noise_realization_count": noise_count,
                **{
                    f"residual_advantage_{metric}": float(
                        np.mean([float(row[f"residual_advantage_{metric}"]) for row in members])
                    )
                    for metric in _MOTION_METRICS
                },
            }
        )
    sample_ids = {str(row["sample_id"]) for row in sample_rows}
    if len(sample_ids) != settings.expected_test_sample_count:
        raise ValueError(
            f"expected {settings.expected_test_sample_count} E7 motion samples; "
            f"found {len(sample_ids)}"
        )
    ci_rows: list[dict[str, Any]] = []
    for condition_index, (channel_uses, snr_db) in enumerate(sorted(conditions)):
        members = [
            row
            for row in sample_rows
            if int(row["channel_uses"]) == channel_uses and float(row["snr_db"]) == snr_db
        ]
        if len(members) != settings.expected_test_sample_count:
            raise ValueError("E7 motion condition has an incomplete sample set")
        for metric_index, metric in enumerate(_MOTION_METRICS):
            seed = settings.bootstrap_seed + condition_index * 101 + metric_index
            result = paired_bootstrap_mean(
                [float(row[f"residual_advantage_{metric}"]) for row in members],
                resamples=settings.bootstrap_resamples,
                confidence_level=settings.confidence_level,
                seed=seed,
            )
            ci_rows.append(
                {
                    "channel_uses": channel_uses,
                    "snr_db": snr_db,
                    "metric": metric,
                    "advantage_definition": "full_motion_error_minus_residual_error",
                    **result,
                }
            )
    return sample_rows, ci_rows


def summarize_video_pairs(
    residual_sample_dir: Path,
    full_motion_sample_dir: Path,
    settings: ThesisEvidenceSettings,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Match frozen per-sample video metrics and bootstrap paired errors."""

    residual_files = {path.stem: path for path in residual_sample_dir.glob("*.json")}
    full_files = {path.stem: path for path in full_motion_sample_dir.glob("*.json")}
    if set(residual_files) != set(full_files):
        raise ValueError("residual and full-motion video sample identities differ")
    if len(residual_files) != settings.expected_test_sample_count:
        raise ValueError(
            f"expected {settings.expected_test_sample_count} video samples; "
            f"found {len(residual_files)}"
        )
    paired_rows: list[dict[str, Any]] = []
    for sample_id in sorted(residual_files):
        residual = _video_jscc_index(_read_json(residual_files[sample_id]))
        full = _video_jscc_index(_read_json(full_files[sample_id]))
        if set(residual) != set(full):
            raise ValueError(f"video condition grid differs for {sample_id}")
        for channel_uses, snr_db in sorted(residual):
            residual_row = residual[(channel_uses, snr_db)]
            full_row = full[(channel_uses, snr_db)]
            if int(residual_row["noise_seed"]) != int(full_row["noise_seed"]):
                raise ValueError(f"nominal video noise seed differs for {sample_id}")
            paired_rows.append(
                {
                    "sample_id": sample_id,
                    "channel_uses": channel_uses,
                    "snr_db": snr_db,
                    "noise_seed": int(residual_row["noise_seed"]),
                    **{
                        f"residual_{metric}": float(residual_row[metric])
                        for metric in _VIDEO_METRICS
                    },
                    **{
                        f"full_motion_{metric}": float(full_row[metric])
                        for metric in _VIDEO_METRICS
                    },
                    **{
                        f"residual_advantage_{metric}": (
                            float(full_row[metric]) - float(residual_row[metric])
                        )
                        for metric in _VIDEO_METRICS
                    },
                }
            )
    ci_rows: list[dict[str, Any]] = []
    conditions = sorted({(int(row["channel_uses"]), float(row["snr_db"])) for row in paired_rows})
    for condition_index, (channel_uses, snr_db) in enumerate(conditions):
        members = [
            row
            for row in paired_rows
            if int(row["channel_uses"]) == channel_uses and float(row["snr_db"]) == snr_db
        ]
        if len(members) != settings.expected_test_sample_count:
            raise ValueError("E7 video condition has an incomplete sample set")
        for metric_index, metric in enumerate(_VIDEO_METRICS):
            seed = settings.bootstrap_seed + 10000 + condition_index * 101 + metric_index
            result = paired_bootstrap_mean(
                [float(row[f"residual_advantage_{metric}"]) for row in members],
                resamples=settings.bootstrap_resamples,
                confidence_level=settings.confidence_level,
                seed=seed,
            )
            ci_rows.append(
                {
                    "channel_uses": channel_uses,
                    "snr_db": snr_db,
                    "metric": metric,
                    "advantage_definition": "full_motion_error_minus_residual_error",
                    **result,
                }
            )
    return paired_rows, ci_rows


def run_thesis_evidence(
    settings: ThesisEvidenceSettings,
    sources: ThesisSourceRuns,
    *,
    run_directory: Path | None = None,
    resume: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Generate figures, tables, paired statistics, and a results draft."""

    sources = sources.resolved()
    provenance = _source_provenance(sources)
    fingerprint = config_fingerprint(
        {"thesis_evidence": settings.config, "source_provenance": provenance}
    )
    run_dir = (
        run_directory.resolve()
        if run_directory is not None
        else _new_run_directory(settings.output_root)
    )
    if run_dir.exists():
        if not resume:
            raise FileExistsError(f"thesis evidence pack exists: {run_dir}")
        complete = _read_json(run_dir / "complete.json")
        if complete.get("evidence_fingerprint") != fingerprint:
            raise ValueError("thesis evidence fingerprint mismatch")
        return run_dir, _read_json(run_dir / "summary.json")
    if resume:
        raise FileNotFoundError(f"cannot resume missing thesis evidence pack: {run_dir}")

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = run_dir.with_name(f".{run_dir.name}.{uuid.uuid4().hex}.tmp")
    staging.mkdir()
    try:
        e3_summary = _read_json(sources.e3 / "summary.json")
        e3_video = _read_json(sources.e3 / "reconstruction/summary.json")
        e4_summary = _read_json(sources.e4 / "summary.json")
        e4_video = _read_json(sources.e4 / "reconstruction/summary.json")
        gate_summary = _read_json(sources.gate / "test_summary.json")
        scorer_summary = _read_json(sources.scorer / "evaluation_summary.json")
        comparison_motion = _read_json(sources.comparison / "motion_summary.json")
        comparison_video = _read_json(sources.comparison / "video_summary.json")
        residual_video = _read_json(sources.residual_jscc / "video_reconstruction/summary.json")
        methodological_checks = _methodological_checks(
            _read_json(sources.communication / "summary.json"),
            _read_json(sources.scorer_ablation / "audit_summary.json"),
        )

        audio_rows = _audio_baseline_rows(e3_summary, e3_video)
        oracle_rows = _oracle_residual_rows(e4_summary, e4_video)
        main_rows = _main_comparison_rows(
            gate_summary,
            comparison_motion,
            comparison_video,
            residual_video,
        )
        scorer_rows = _scorer_rows(scorer_summary)
        motion_sample_rows, motion_ci_rows = summarize_motion_pairs(
            _read_jsonl(sources.comparison / "motion_pairs.jsonl"),
            settings,
        )
        video_pair_rows, video_ci_rows = summarize_video_pairs(
            sources.residual_jscc / "video_reconstruction/samples",
            sources.full_motion / "video_reconstruction/samples",
            settings,
        )
        qualitative_rows = _qualitative_rows(
            sources,
            settings,
            sorted({str(row["sample_id"]) for row in video_pair_rows}),
        )
        figure_manifest = _figure_manifest()

        atomic_write_json(staging / "resolved_config.json", dict(settings.config))
        atomic_write_json(staging / "source_provenance.json", provenance)
        atomic_write_json(staging / "environment.json", _environment())
        atomic_write_json(
            staging / "run_metadata.json",
            {
                "evidence_fingerprint": fingerprint,
                "git_commit": _git_commit(),
                "model_training_performed": False,
                "model_selection_performed": False,
                "test_metrics_recomputed": False,
                "bootstrap_unit": "sample_after_within_sample_noise_mean",
            },
        )
        _write_rows(staging, "audio_baseline", audio_rows)
        _write_rows(staging, "oracle_residual_curve", oracle_rows)
        _write_rows(staging, "main_comparison", main_rows)
        _write_rows(staging, "scorer_summary", scorer_rows)
        _write_rows(staging, "e7_motion_sample_differences", motion_sample_rows)
        _write_rows(staging, "e7_motion_bootstrap", motion_ci_rows)
        _write_rows(staging, "e7_video_sample_differences", video_pair_rows)
        _write_rows(staging, "e7_video_bootstrap", video_ci_rows)
        atomic_write_json(staging / "qualitative_selection.json", {"rows": qualitative_rows})
        atomic_write_json(staging / "figure_manifest.json", {"figures": figure_manifest})
        atomic_write_json(staging / "methodological_checks.json", methodological_checks)
        _write_figures(
            staging / "figures",
            audio_rows,
            oracle_rows,
            main_rows,
            gate_summary,
            motion_ci_rows,
            settings.figure_dpi,
        )

        summary = _summary(
            settings,
            fingerprint,
            provenance,
            main_rows,
            motion_ci_rows,
            video_ci_rows,
            figure_manifest,
            qualitative_rows,
        )
        atomic_write_json(staging / "summary.json", summary)
        _atomic_write_text(
            staging / "results_chapter_draft.md",
            _results_chapter(summary, main_rows, motion_ci_rows, video_ci_rows),
        )
        atomic_write_json(
            staging / "complete.json",
            {
                "status": "complete",
                "evidence_fingerprint": fingerprint,
                "main_table_row_count": len(main_rows),
                "motion_bootstrap_row_count": len(motion_ci_rows),
                "video_bootstrap_row_count": len(video_ci_rows),
                "figure_count": len(figure_manifest),
            },
        )
        os.replace(staging, run_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return run_dir, summary


def _audio_baseline_rows(
    summary: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
) -> list[dict[str, Any]]:
    groups = summary.get("groups")
    video_groups = reconstruction.get("groups")
    if not isinstance(groups, list) or not isinstance(video_groups, list):
        raise ValueError("E3 summary is incomplete")
    complete = {
        (str(row["method"]), row.get("seed")): row for row in groups if row.get("split") == "test"
    }
    validation_gru = [
        row
        for row in groups
        if row.get("split") == "validation" and row.get("method") == "audio_gru"
    ]
    selected_seed = int(min(validation_gru, key=lambda row: float(row["l1"]))["seed"])
    video_index = {
        (str(row["method"]), row.get("seed")): row
        for row in video_groups
        if row.get("split") == "test"
    }
    output: list[dict[str, Any]] = []
    for method in ("train_mean", "zero_motion", "audio_gru", "oracle_persistence"):
        seed: int | None = selected_seed if method == "audio_gru" else None
        motion = complete[(method, seed)]
        video = video_index[(method, seed)]
        output.append(
            {
                "method": method,
                "model_seed": seed,
                "selected_on": "validation_l1" if method == "audio_gru" else "fixed_baseline",
                "sample_count": int(motion["sample_count"]),
                "motion_l1": float(motion["l1"]),
                "motion_rmse": float(motion["rmse"]),
                "motion_velocity_l1": float(motion["velocity_l1"]),
                "mouth_mae": float(video["oracle_mouth_mae"]),
                "mouth_nme": float(video["oracle_mouth_nme"]),
                "psnr_db": float(video["oracle_psnr_db"]),
                "ssim": float(video["oracle_ssim"]),
            }
        )
    return output


def _oracle_residual_rows(
    summary: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
) -> list[dict[str, Any]]:
    groups = summary.get("groups")
    video_groups = reconstruction.get("groups")
    if not isinstance(groups, list) or not isinstance(video_groups, list):
        raise ValueError("E4 summary is incomplete")
    motion: dict[int, Mapping[str, Any]] = {}
    for row in groups:
        if row.get("split") != "test":
            continue
        condition = str(row["condition"])
        if condition == "prediction_only":
            motion[0] = row
        elif condition == "magnitude_top_k" and row.get("selection_space") == "raw":
            motion[int(row["k"])] = row
        elif condition == "full_residual_oracle":
            motion[18] = row
    video: dict[int, Mapping[str, Any]] = {}
    for row in video_groups:
        if row.get("split") != "test":
            continue
        condition = str(row["condition"])
        if condition == "prediction_only":
            video[0] = row
        elif condition == "magnitude_top_k" and row.get("selection_space") == "raw":
            video[int(row["k"])] = row
        elif condition in {"full_residual_oracle", "dense_motion_oracle"}:
            video[18] = row
    if set(motion) != {0, 1, 2, 4, 6, 9, 12, 18}:
        raise ValueError("E4 motion curve does not contain the frozen K grid")
    return [
        {
            "k": k,
            "keep_ratio": k / 18.0,
            "motion_l1": float(motion[k]["l1"]),
            "motion_rmse": float(motion[k]["rmse"]),
            "motion_velocity_l1": float(motion[k]["velocity_l1"]),
            "raw_energy_retained_fraction": float(motion[k]["raw_energy_retained_fraction"]),
            "mouth_mae": None if k not in video else float(video[k]["oracle_mouth_mae"]),
            "mouth_nme": None if k not in video else float(video[k]["oracle_mouth_nme"]),
        }
        for k in sorted(motion)
    ]


def _main_comparison_rows(
    gate_summary: Mapping[str, Any],
    motion_summary: Mapping[str, Any],
    video_summary: Mapping[str, Any],
    residual_video_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    gate_groups = gate_summary.get("groups")
    motion_groups = motion_summary.get("rows")
    video_groups = video_summary.get("rows")
    residual_groups = residual_video_summary.get("groups")
    source_groups = (gate_groups, motion_groups, video_groups, residual_groups)
    if not all(isinstance(value, list) for value in source_groups):
        raise ValueError("main comparison source is incomplete")
    gate_index = {(int(row["channel_uses"]), float(row["snr_db"])): row for row in gate_groups}
    motion_index = {(int(row["channel_uses"]), float(row["snr_db"])): row for row in motion_groups}
    video_index = {(int(row["channel_uses"]), float(row["snr_db"])): row for row in video_groups}
    prediction_candidates = [
        row for row in residual_groups if row.get("family") == "prediction_only"
    ]
    if len(prediction_candidates) != 1:
        raise ValueError("residual video summary must have one prediction-only group")
    prediction_nme = float(prediction_candidates[0]["oracle_mouth_nme"])
    prediction_mae = float(prediction_candidates[0]["oracle_mouth_mae"])
    expected = {(c, snr) for c in _CHANNEL_USES for snr in _TEST_SNR_DB}
    if set(gate_index) != expected or set(motion_index) != expected or set(video_index) != expected:
        raise ValueError("main comparison C x SNR grids differ")
    output: list[dict[str, Any]] = []
    for channel_uses, snr_db in sorted(expected):
        gate = gate_index[(channel_uses, snr_db)]
        motion = motion_index[(channel_uses, snr_db)]
        video = video_index[(channel_uses, snr_db)]
        if abs(float(gate["always_send_l1"]) - float(motion["residual_l1"])) > 1e-12:
            raise ValueError("gate residual L1 differs from E7 frozen comparison")
        transmit = str(gate["decision"]) == "send_jscc"
        residual_nme = float(video["residual_oracle_mouth_nme"])
        residual_mae = float(video["residual_oracle_mouth_mae"])
        output.append(
            {
                "channel_uses": channel_uses,
                "snr_db": snr_db,
                "complex_symbols_per_clip": int(motion["complex_symbols_per_clip"]),
                "complex_symbols_per_second": float(motion["complex_symbols_per_second"]),
                "gate_transmit": transmit,
                "gated_symbols_per_clip": (
                    int(motion["complex_symbols_per_clip"]) if transmit else 0
                ),
                "prediction_motion_l1": float(gate["prediction_l1"]),
                "full_motion_l1": float(motion["full_motion_l1"]),
                "residual_motion_l1": float(motion["residual_l1"]),
                "gated_residual_motion_l1": float(gate["gated_l1"]),
                "prediction_mouth_mae": prediction_mae,
                "full_motion_mouth_mae": float(video["full_motion_oracle_mouth_mae"]),
                "residual_mouth_mae": residual_mae,
                "gated_residual_mouth_mae": residual_mae if transmit else prediction_mae,
                "prediction_mouth_nme": prediction_nme,
                "full_motion_mouth_nme": float(video["full_motion_oracle_mouth_nme"]),
                "residual_mouth_nme": residual_nme,
                "gated_residual_mouth_nme": residual_nme if transmit else prediction_nme,
            }
        )
    return output


def _scorer_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    aggregate = summary.get("aggregate")
    if not isinstance(aggregate, list):
        raise ValueError("scorer summary is incomplete")
    output: list[dict[str, Any]] = []
    for channel_uses in _CHANNEL_USES:
        snrs = (0.0, 5.0, 10.0)
        index = {
            (str(row["method"]), float(row["snr_db"])): row
            for row in aggregate
            if int(row["channel_uses"]) == channel_uses
            and float(row["snr_db"]) in snrs
            and str(row["method"]) in {"dense_jscc", "raw_magnitude", "learned_scorer"}
        }
        if len(index) != 9:
            raise ValueError(f"scorer aggregate is incomplete for C={channel_uses}")
        raw = np.mean([float(index[("raw_magnitude", snr)]["l1_mean"]) for snr in snrs])
        learned = np.mean([float(index[("learned_scorer", snr)]["l1_mean"]) for snr in snrs])
        dense = np.mean([float(index[("dense_jscc", snr)]["l1_mean"]) for snr in snrs])
        output.append(
            {
                "channel_uses": channel_uses,
                "k": 2 * channel_uses,
                "snr_average_db": "0,5,10",
                "dense_l1": float(dense),
                "raw_magnitude_l1": float(raw),
                "learned_scorer_l1": float(learned),
                "learned_advantage_vs_raw_percent": float((raw - learned) / raw * 100.0),
                "dense_advantage_vs_learned_percent": float((learned - dense) / learned * 100.0),
            }
        )
    return output


def _video_jscc_index(payload: Mapping[str, Any]) -> dict[tuple[int, float], Mapping[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("video sample JSON has no rows")
    output = {
        (int(row["channel_uses"]), float(row["snr_db"])): row
        for row in rows
        if row.get("family") == "jscc_awgn"
    }
    expected = {(c, snr) for c in _CHANNEL_USES for snr in _TEST_SNR_DB}
    if set(output) != expected:
        raise ValueError("video sample does not contain the frozen C x SNR grid")
    return output


def _qualitative_rows(
    sources: ThesisSourceRuns,
    settings: ThesisEvidenceSettings,
    sample_ids: Sequence[str],
) -> list[dict[str, Any]]:
    if len(sample_ids) != settings.expected_test_sample_count:
        raise ValueError("qualitative selection sample count differs")
    rows: list[dict[str, Any]] = []
    for position in settings.qualitative_positions:
        sample_id = sample_ids[position]
        residual_media = sources.residual_jscc / "video_reconstruction/media/test" / sample_id
        full_media = sources.full_motion / "video_reconstruction/media/test" / sample_id
        if not residual_media.is_dir() or not full_media.is_dir():
            raise ValueError(f"representative media is missing for {sample_id}")
        rows.append(
            {
                "sort_position": position,
                "sample_id": sample_id,
                "selection_rule": "fixed_lexicographic_position",
                "residual_media_relative_path": (f"video_reconstruction/media/test/{sample_id}"),
                "full_motion_media_relative_path": (f"video_reconstruction/media/test/{sample_id}"),
                "recommended_channel_uses": 4,
                "recommended_snr_db": [-5.0, 5.0, 10.0],
            }
        )
    return rows


def _source_provenance(sources: ThesisSourceRuns) -> dict[str, Any]:
    required = {
        "e3_summary_sha256": sources.e3 / "summary.json",
        "e3_video_summary_sha256": sources.e3 / "reconstruction/summary.json",
        "e4_summary_sha256": sources.e4 / "summary.json",
        "e4_video_summary_sha256": sources.e4 / "reconstruction/summary.json",
        "e5_evaluation_complete_sha256": sources.residual_jscc / "evaluation_complete.json",
        "e5_video_summary_sha256": (sources.residual_jscc / "video_reconstruction/summary.json"),
        "gate_summary_sha256": sources.gate / "test_summary.json",
        "scorer_summary_sha256": sources.scorer / "evaluation_summary.json",
        "scorer_ablation_summary_sha256": sources.scorer_ablation / "audit_summary.json",
        "communication_summary_sha256": sources.communication / "summary.json",
        "full_motion_evaluation_complete_sha256": (
            sources.full_motion / "evaluation_complete.json"
        ),
        "comparison_motion_pairs_sha256": sources.comparison / "motion_pairs.jsonl",
        "comparison_motion_summary_sha256": sources.comparison / "motion_summary.json",
        "comparison_video_summary_sha256": sources.comparison / "video_summary.json",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen thesis source: {path}")
    marker_paths = (
        sources.e4 / "analysis_complete.json",
        sources.residual_jscc / "evaluation_complete.json",
        sources.residual_jscc / "video_reconstruction/complete.json",
        sources.gate / "complete.json",
        sources.scorer / "evaluation_complete.json",
        sources.scorer_ablation / "audit_complete.json",
        sources.communication / "complete.json",
        sources.full_motion / "evaluation_complete.json",
        sources.full_motion / "video_reconstruction/complete.json",
        sources.comparison / "complete.json",
    )
    for marker_path in marker_paths:
        marker = _read_json(marker_path)
        if marker.get("status", "complete") != "complete":
            raise ValueError(f"frozen thesis source is incomplete: {marker_path}")
    return {
        "evidence_generator_sha256": file_sha256(Path(__file__).resolve()),
        **{name: file_sha256(path) for name, path in required.items()},
        "residual_video_sample_tree_sha256": _json_tree_sha256(
            sources.residual_jscc / "video_reconstruction/samples"
        ),
        "full_motion_video_sample_tree_sha256": _json_tree_sha256(
            sources.full_motion / "video_reconstruction/samples"
        ),
    }


def _methodological_checks(
    communication: Mapping[str, Any],
    scorer_ablation: Mapping[str, Any],
) -> dict[str, Any]:
    if communication.get("status") != "complete":
        raise ValueError("communication report is incomplete")
    if communication.get("digital_bitrate_defined") is not False:
        raise ValueError("communication report must not claim a digital bitrate")
    if communication.get("rate_unit") != "complex_channel_symbols":
        raise ValueError("communication report uses an unexpected rate unit")
    if scorer_ablation.get("status") != "complete":
        raise ValueError("scorer ablation is incomplete")
    if scorer_ablation.get("test_data_accessed") is not False:
        raise ValueError("scorer ablation must remain validation-only")
    return {
        "communication": {
            "digital_bitrate_defined": False,
            "rate_unit": "complex_channel_symbols",
            "eligible_frame_count": int(communication["eligible_frame_count"]),
            "motion_dimension": int(communication["motion_dimension"]),
            "all_transmitted_sparse_points_dominated_by_dense_same_rate": bool(
                communication["all_transmitted_sparse_points_dominated_by_dense_same_rate"]
            ),
            "unmeasured_costs": list(communication["unmeasured_costs"]),
        },
        "scorer_ablation": {
            "evaluation_scope": str(scorer_ablation["evaluation_scope"]),
            "audit_sample_count": int(scorer_ablation["audit_sample_count"]),
            "audit_speakers": list(scorer_ablation["audit_speakers"]),
            "test_data_accessed": False,
        },
    }


def _json_tree_sha256(path: Path) -> str:
    files = sorted(path.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no JSON sample files in {path}")
    digest = hashlib.sha256()
    for file_path in files:
        digest.update(file_path.name.encode())
        digest.update(file_sha256(file_path).encode())
    return digest.hexdigest()


def _write_rows(path: Path, stem: str, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty evidence table: {stem}")
    atomic_write_json(path / f"{stem}.json", {"rows": list(rows)})
    _write_dict_csv(path / f"{stem}.csv", rows)


def _atomic_write_text(path: Path, value: str) -> None:
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
        handle.write(value)
    os.replace(temporary, path)


def _figure_manifest() -> list[dict[str, Any]]:
    return [
        {
            "figure_id": "F1",
            "filename": "01_audio_prediction_baseline.png",
            "title": "Audio-to-motion baseline on the identity-isolated test speaker",
            "claim": "The validation-selected causal GRU outperforms static baselines.",
            "conditions": "Ten-speaker 8/1/1 identity-isolated GRID split; test speaker s7.",
            "metric_direction": "Lower motion L1 and mouth NME are better.",
            "boundary": "Oracle persistence is non-deployable and is shown only as a bound.",
        },
        {
            "figure_id": "F2",
            "filename": "02_oracle_residual_curve.png",
            "title": "Oracle residual rate-quality upper bound",
            "claim": (
                "A small number of true high-magnitude residual dimensions "
                "carries substantial value."
            ),
            "conditions": "Frozen E3 prediction and sender-known true residual; no channel.",
            "metric_direction": "Lower motion L1 and mouth NME are better.",
            "boundary": "This oracle curve is not a deployable selector or bitrate result.",
        },
        {
            "figure_id": "F3",
            "filename": "03_residual_jscc_motion_vs_snr.png",
            "title": "Residual JSCC motion error versus SNR",
            "claim": (
                "Residual transmission helps at 0/5/10 dB and degrades outside training at -5 dB."
            ),
            "conditions": "Sionna complex AWGN; C=1/2/3/4 symbols per eligible frame.",
            "metric_direction": "Lower motion L1 is better.",
            "boundary": "C is a complex-symbol budget, not bit/s.",
        },
        {
            "figure_id": "F4",
            "filename": "04_residual_jscc_video_vs_snr.png",
            "title": "Residual JSCC mouth NME versus SNR",
            "claim": "Motion-space gains generally transfer to frozen mouth reconstruction.",
            "conditions": "Frozen LivePortrait reconstruction of 100 s7 test clips.",
            "metric_direction": "Lower mouth NME is better.",
            "boundary": "NME is measured against lip-only oracle reconstruction.",
        },
        {
            "figure_id": "F5",
            "filename": "05_validation_gate_safety.png",
            "title": "Validation-frozen low-SNR safety gate",
            "claim": "The gate removes -5 dB degradation without changing higher-SNR gains.",
            "conditions": "Threshold selected only on s3 validation and frozen before s7 test.",
            "metric_direction": "Positive relative L1 improvement is better.",
            "boundary": "The gate uses global SNR and is not a content-aware residual selector.",
        },
        {
            "figure_id": "F6",
            "filename": "06_full_motion_vs_residual_heatmap.png",
            "title": "Matched-budget residual advantage over full-motion JSCC",
            "claim": "Residual gains concentrate at lower SNR and tighter budgets.",
            "conditions": (
                "Matched architecture, complex-symbol budget, AWGN grid, and test samples."
            ),
            "metric_direction": "Positive advantage means full-motion error minus residual error.",
            "boundary": "Point estimates do not imply residual wins at every condition.",
        },
        {
            "figure_id": "A1",
            "filename": "07_e7_paired_bootstrap_l1.png",
            "title": "Sample-level paired bootstrap intervals for E7 motion L1",
            "claim": (
                "Uncertainty is computed across 100 samples after averaging noise within sample."
            ),
            "conditions": "Three nominal noise seeds per motion sample and 10,000 resamples.",
            "metric_direction": (
                "Intervals above zero favor residual; below zero favor full-motion."
            ),
            "boundary": (
                "Intervals are pointwise and descriptive; no multiplicity correction is used."
            ),
        },
    ]


def _write_figures(
    path: Path,
    audio_rows: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[Mapping[str, Any]],
    main_rows: Sequence[Mapping[str, Any]],
    gate_summary: Mapping[str, Any],
    motion_ci_rows: Sequence[Mapping[str, Any]],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    path.mkdir(parents=True, exist_ok=True)
    colors = {
        "prediction": "#4d4d4d",
        "full": "#e67e22",
        "residual": "#1f77b4",
        "gated": "#2ca02c",
    }
    labels = [str(row["method"]).replace("_", " ") for row in audio_rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(labels, [float(row["motion_l1"]) for row in audio_rows], color="#4c78a8")
    axes[0].set_ylabel("Motion L1 (lower is better)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(labels, [float(row["mouth_nme"]) for row in audio_rows], color="#72b7b2")
    axes[1].set_ylabel("Mouth NME (lower is better)")
    axes[1].tick_params(axis="x", rotation=25)
    figure.suptitle("Identity-isolated audio prediction baseline")
    figure.tight_layout()
    figure.savefig(path / "01_audio_prediction_baseline.png", dpi=dpi)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(
        [int(row["k"]) for row in oracle_rows],
        [float(row["motion_l1"]) for row in oracle_rows],
        marker="o",
        color=colors["residual"],
    )
    nme_rows = [row for row in oracle_rows if row["mouth_nme"] is not None]
    axes[1].plot(
        [int(row["k"]) for row in nme_rows],
        [float(row["mouth_nme"]) for row in nme_rows],
        marker="o",
        color="#72b7b2",
    )
    axes[0].set_ylabel("Motion L1")
    axes[1].set_ylabel("Mouth NME")
    for axis in axes:
        axis.set_xlabel("Retained residual dimensions K / 18")
        axis.grid(alpha=0.3)
    figure.suptitle("Oracle true-residual upper bound")
    figure.tight_layout()
    figure.savefig(path / "02_oracle_residual_curve.png", dpi=dpi)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    prediction = float(main_rows[0]["prediction_motion_l1"])
    axis.axhline(prediction, color=colors["prediction"], linestyle="--", label="prediction only")
    for channel_uses in _CHANNEL_USES:
        members = [row for row in main_rows if int(row["channel_uses"]) == channel_uses]
        axis.plot(
            [float(row["snr_db"]) for row in members],
            [float(row["residual_motion_l1"]) for row in members],
            marker="o",
            label=f"residual C={channel_uses}",
        )
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("Motion L1")
    axis.set_title("Residual JSCC motion quality")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path / "03_residual_jscc_motion_vs_snr.png", dpi=dpi)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    prediction_nme = float(main_rows[0]["prediction_mouth_nme"])
    axis.axhline(
        prediction_nme,
        color=colors["prediction"],
        linestyle="--",
        label="prediction only",
    )
    for channel_uses in _CHANNEL_USES:
        members = [row for row in main_rows if int(row["channel_uses"]) == channel_uses]
        axis.plot(
            [float(row["snr_db"]) for row in members],
            [float(row["residual_mouth_nme"]) for row in members],
            marker="o",
            label=f"residual C={channel_uses}",
        )
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("Mouth NME")
    axis.set_title("Frozen LivePortrait mouth quality")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path / "04_residual_jscc_video_vs_snr.png", dpi=dpi)
    plt.close(figure)

    gate_groups = gate_summary["groups"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for channel_uses in _CHANNEL_USES:
        members = [row for row in gate_groups if int(row["channel_uses"]) == channel_uses]
        x = [float(row["snr_db"]) for row in members]
        axes[0].plot(
            x,
            [
                float(row["always_send_relative_l1_improvement_vs_prediction"]) * 100.0
                for row in members
            ],
            marker="o",
            label=f"C={channel_uses}",
        )
        axes[1].plot(
            x,
            [float(row["gated_relative_l1_improvement_vs_prediction"]) * 100.0 for row in members],
            marker="o",
            label=f"C={channel_uses}",
        )
    axes[0].set_title("Always send")
    axes[1].set_title("Validation-frozen gate")
    for axis in axes:
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xlabel("SNR (dB)")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("L1 improvement vs prediction (%)")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path / "05_validation_gate_safety.png", dpi=dpi)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for axis, key, title in (
        (axes[0], "residual_motion_l1", "Motion L1 advantage (%)"),
        (axes[1], "residual_mouth_nme", "Mouth NME advantage (%)"),
    ):
        matrix = np.zeros((len(_TEST_SNR_DB), len(_CHANNEL_USES)), dtype=np.float64)
        for row in main_rows:
            snr_index = _TEST_SNR_DB.index(float(row["snr_db"]))
            c_index = _CHANNEL_USES.index(int(row["channel_uses"]))
            full_key = "full_motion_l1" if key == "residual_motion_l1" else "full_motion_mouth_nme"
            matrix[snr_index, c_index] = (
                (float(row[full_key]) - float(row[key])) / float(row[full_key]) * 100.0
            )
        limit = max(abs(float(matrix.min())), abs(float(matrix.max())))
        image = axis.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        axis.set_xticks(range(len(_CHANNEL_USES)), [f"C={value}" for value in _CHANNEL_USES])
        axis.set_yticks(range(len(_TEST_SNR_DB)), [f"{value:g} dB" for value in _TEST_SNR_DB])
        axis.set_title(title)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.suptitle("Residual advantage over full-motion JSCC")
    figure.tight_layout()
    figure.savefig(path / "06_full_motion_vs_residual_heatmap.png", dpi=dpi)
    plt.close(figure)

    l1_rows = [row for row in motion_ci_rows if row["metric"] == "l1"]
    labels = [f"C={int(row['channel_uses'])}, {float(row['snr_db']):g}dB" for row in l1_rows]
    points = np.asarray([float(row["point_estimate"]) for row in l1_rows])
    lower = np.asarray([float(row["ci_lower"]) for row in l1_rows])
    upper = np.asarray([float(row["ci_upper"]) for row in l1_rows])
    positions = np.arange(len(l1_rows))
    figure, axis = plt.subplots(figsize=(8.5, 6.2))
    axis.errorbar(
        points,
        positions,
        xerr=np.vstack((points - lower, upper - points)),
        fmt="o",
        color=colors["residual"],
        capsize=3,
    )
    axis.axvline(0, color="black", linestyle="--", linewidth=0.9)
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Full-motion L1 minus residual L1")
    axis.set_title("Sample-level paired bootstrap intervals")
    axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path / "07_e7_paired_bootstrap_l1.png", dpi=dpi)
    plt.close(figure)


def _summary(
    settings: ThesisEvidenceSettings,
    fingerprint: str,
    provenance: Mapping[str, Any],
    main_rows: Sequence[Mapping[str, Any]],
    motion_ci_rows: Sequence[Mapping[str, Any]],
    video_ci_rows: Sequence[Mapping[str, Any]],
    figure_manifest: Sequence[Mapping[str, Any]],
    qualitative_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    motion_l1 = [row for row in motion_ci_rows if row["metric"] == "l1"]
    video_nme = [row for row in video_ci_rows if row["metric"] == "oracle_mouth_nme"]
    return {
        "schema_version": 1,
        "status": "complete",
        "evidence_fingerprint": fingerprint,
        "scope": "read_only_frozen_e3_to_e7_thesis_reporting",
        "model_training_performed": False,
        "model_selection_performed": False,
        "test_metrics_recomputed": False,
        "digital_bitrate_defined": False,
        "rate_unit": "complex_channel_symbols",
        "bootstrap_unit": "test_sample_after_within_sample_noise_mean",
        "bootstrap_seed": settings.bootstrap_seed,
        "bootstrap_resamples": settings.bootstrap_resamples,
        "confidence_level": settings.confidence_level,
        "interval_role": "descriptive_pointwise",
        "multiplicity_adjusted": False,
        "test_sample_count": settings.expected_test_sample_count,
        "main_table_row_count": len(main_rows),
        "motion_bootstrap_row_count": len(motion_ci_rows),
        "video_bootstrap_row_count": len(video_ci_rows),
        "figure_count": len(figure_manifest),
        "qualitative_sample_count": len(qualitative_rows),
        "motion_l1_residual_point_win_count": sum(
            float(row["point_estimate"]) > 0 for row in motion_l1
        ),
        "motion_l1_residual_ci_win_count": sum(
            row["direction"] == "residual_better" for row in motion_l1
        ),
        "motion_l1_full_motion_ci_win_count": sum(
            row["direction"] == "full_motion_better" for row in motion_l1
        ),
        "motion_l1_inconclusive_ci_count": sum(
            row["direction"] == "inconclusive" for row in motion_l1
        ),
        "mouth_nme_residual_point_win_count": sum(
            float(row["point_estimate"]) > 0 for row in video_nme
        ),
        "mouth_nme_residual_ci_win_count": sum(
            row["direction"] == "residual_better" for row in video_nme
        ),
        "mouth_nme_full_motion_ci_win_count": sum(
            row["direction"] == "full_motion_better" for row in video_nme
        ),
        "mouth_nme_inconclusive_ci_count": sum(
            row["direction"] == "inconclusive" for row in video_nme
        ),
        "source_provenance": dict(provenance),
    }


def _results_chapter(
    summary: Mapping[str, Any],
    main_rows: Sequence[Mapping[str, Any]],
    motion_ci_rows: Sequence[Mapping[str, Any]],
    video_ci_rows: Sequence[Mapping[str, Any]],
) -> str:
    strongest = max(
        main_rows,
        key=lambda row: (
            (float(row["full_motion_l1"]) - float(row["residual_motion_l1"]))
            / float(row["full_motion_l1"])
        ),
    )
    return f"""# 实验结果与讨论（E8 自动草稿）

## 统计协议

本章只读取冻结 E3–E7 产物，没有重新训练、选择模型或计算新的 test 预测。
E7 运动结果先在每条 test 样本内部对三个名义噪声实现求均值，再以
{summary["test_sample_count"]} 条样本为统计单位执行
{summary["bootstrap_resamples"]} 次固定种子配对 bootstrap，报告
{float(summary["confidence_level"]) * 100:.0f}% 百分位区间。视频结果使用两条
链路共同的名义噪声种子 42，以相同样本、C 和 SNR 配对。

## 核心结果

残差 JSCC 在运动 L1 的 16 个 C×SNR 点估计中胜出
{summary["motion_l1_residual_point_win_count"]} 个；置信区间完全支持 residual
的有 {summary["motion_l1_residual_ci_win_count"]} 个，完全支持 full-motion 的有
{summary["motion_l1_full_motion_ci_win_count"]} 个，其余
{summary["motion_l1_inconclusive_ci_count"]} 个区间跨零。嘴部 NME 点估计中
residual 胜出 {summary["mouth_nme_residual_point_win_count"]} 个；置信区间完全
支持 residual/full-motion 的分别为
{summary["mouth_nme_residual_ci_win_count"]}/
{summary["mouth_nme_full_motion_ci_win_count"]} 个，另有
{summary["mouth_nme_inconclusive_ci_count"]} 个区间跨零。

运动 L1 相对优势最大的条件为 C={int(strongest["channel_uses"])}、
SNR={float(strongest["snr_db"]):g} dB。总体趋势是低/中 SNR 和紧预算下
residual 更有效，高 SNR、大预算时 full-motion 可以追平或局部胜出。这支持
“音频先验在受限视觉信道中减少待恢复不确定性”的有条件结论，不支持
“残差在所有条件下始终最优”。

## 结果边界

当前横轴是复信道符号，不是数字 bit/s；没有计入音频链路、参考脸、量化、调制
编码、索引和协议开销。GRID 只包含受控语料和单一 test 身份，结论不能直接外推
到真实会议或跨数据集。名义噪声种子匹配，但 validation 选中的模型种子不同时，
精确逐元素噪声实现不完全相同。
16 个 C×SNR 区间是逐条件的描述性区间，没有做多重比较校正，因此不应将
“区间不跨零”的计数解释为控制族错误率后的验证性显著性结论。

## 表格索引

- 主比较：`main_comparison.csv`
- 运动样本差值：`e7_motion_sample_differences.csv`
- 运动 bootstrap：`e7_motion_bootstrap.csv`
- 视频样本差值：`e7_video_sample_differences.csv`
- 视频 bootstrap：`e7_video_bootstrap.csv`

本草稿由冻结数据自动生成，正式论文写作时应结合图注和方法章节润色，不应手工
修改数值后脱离源哈希。
"""
