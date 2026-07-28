"""Communication-cost and rate-quality report from frozen artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from av_semcom.data.preprocessing import atomic_write_json, config_fingerprint
from av_semcom.models.predictor.artifacts import file_sha256
from av_semcom.models.selection.gate import (
    GatePolicy,
    _environment,
    _git_commit,
    _new_run_directory,
    _read_json,
    _write_dict_csv,
)
from av_semcom.utils.config import ConfigError

_MOTION_METRICS = (
    "l1_mean",
    "rmse_mean",
    "velocity_l1_mean",
    "normalized_residual_mse_mean",
)


@dataclass(frozen=True)
class CommunicationReportSettings:
    """Frozen accounting units for one GRID clip."""

    output_root: Path
    frame_rate: int
    frame_count: int
    reference_frame_count: int
    motion_dimension: int
    methods: tuple[str, ...]
    config: Mapping[str, Any]

    @property
    def eligible_frame_count(self) -> int:
        return self.frame_count - self.reference_frame_count

    @property
    def clip_duration_seconds(self) -> float:
        return self.frame_count / self.frame_rate

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> CommunicationReportSettings:
        """Validate a symbol-domain report that cannot claim digital bitrate."""

        raw = config.get("communication_report")
        if not isinstance(raw, Mapping):
            raise ConfigError("communication_report configuration must be a mapping")
        frame_rate = int(raw.get("frame_rate", 0))
        frame_count = int(raw.get("frame_count", 0))
        reference_count = int(raw.get("reference_frame_count", 0))
        motion_dimension = int(raw.get("motion_dimension", 0))
        if frame_rate <= 0 or frame_count <= 1 or motion_dimension <= 0:
            raise ConfigError("communication report dimensions must be positive")
        if not 0 < reference_count < frame_count:
            raise ConfigError("reference_frame_count must be in [1, frame_count)")
        methods_raw = raw.get("methods")
        if not isinstance(methods_raw, list) or not methods_raw:
            raise ConfigError("communication_report.methods must be a non-empty list")
        methods = tuple(str(value) for value in methods_raw)
        expected_methods = ("dense_jscc", "raw_magnitude", "learned_scorer")
        if methods != expected_methods:
            raise ConfigError(f"communication report methods must equal {expected_methods}")
        forbidden_true = (
            "digital_bitrate_defined",
            "include_audio_side_information_cost",
            "include_reference_face_cost",
        )
        if any(bool(raw.get(key, False)) for key in forbidden_true):
            raise ConfigError(
                "current analog JSCC report cannot define bitrate or include unmeasured costs"
            )
        output_raw = raw.get("output_dir", "outputs/communication_report")
        if not isinstance(output_raw, str) or not output_raw:
            raise ConfigError("communication_report.output_dir must be a path")
        output_root = Path(output_raw)
        if not output_root.is_absolute():
            output_root = Path(__file__).resolve().parents[3] / output_root
        return cls(
            output_root=output_root.resolve(),
            frame_rate=frame_rate,
            frame_count=frame_count,
            reference_frame_count=reference_count,
            motion_dimension=motion_dimension,
            methods=methods,
            config=dict(raw),
        )


def run_communication_report(
    settings: CommunicationReportSettings,
    e5_run_dir: Path,
    gate_run_dir: Path,
    scorer_run_dir: Path,
    ablation_run_dir: Path,
    *,
    run_directory: Path | None = None,
    resume: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Build a descriptive report without model selection or metric recomputation."""

    e5_run_dir = e5_run_dir.resolve()
    gate_run_dir = gate_run_dir.resolve()
    scorer_run_dir = scorer_run_dir.resolve()
    ablation_run_dir = ablation_run_dir.resolve()
    source = _source_provenance(
        e5_run_dir,
        gate_run_dir,
        scorer_run_dir,
        ablation_run_dir,
    )
    fingerprint = config_fingerprint({"communication_report": settings.config, "source": source})
    run_dir = (
        run_directory.resolve()
        if run_directory is not None
        else _new_run_directory(settings.output_root)
    )
    complete_path = run_dir / "complete.json"
    if run_dir.exists():
        if not resume:
            raise FileExistsError(f"communication report exists: {run_dir}")
        complete = _read_json(complete_path)
        if complete.get("report_fingerprint") != fingerprint:
            raise ValueError("communication report fingerprint mismatch")
        return run_dir, _read_json(run_dir / "summary.json")
    if resume:
        raise FileNotFoundError(f"cannot resume missing communication report: {run_dir}")
    run_dir.mkdir(parents=True)

    scorer_summary = _read_json(scorer_run_dir / "evaluation_summary.json")
    gate_summary = _read_json(gate_run_dir / "test_summary.json")
    gate_policy = GatePolicy.from_dict(_read_json(gate_run_dir / "policy.json"))
    motion_rows = _motion_rate_quality(
        settings,
        scorer_summary,
        gate_summary,
        gate_policy,
    )
    video_rows = _video_rate_quality(
        settings,
        _read_json(e5_run_dir / "video_reconstruction/summary.json"),
        gate_policy,
    )
    accounting_rows = _accounting_rows(settings, gate_policy)
    motion_sparse_rows = [
        row
        for row in motion_rows
        if row["method"] in {"raw_magnitude", "learned_scorer"} and row["gate_transmit"]
    ]
    summary = {
        "schema_version": 1,
        "status": "complete",
        "report_fingerprint": fingerprint,
        "scope": "post_hoc_frozen_aggregate_reporting",
        "model_selection_performed": False,
        "test_metrics_recomputed": False,
        "digital_bitrate_defined": False,
        "rate_unit": "complex_channel_symbols",
        "clip_duration_seconds": settings.clip_duration_seconds,
        "eligible_frame_count": settings.eligible_frame_count,
        "motion_dimension": settings.motion_dimension,
        "accounting_row_count": len(accounting_rows),
        "motion_row_count": len(motion_rows),
        "video_row_count": len(video_rows),
        "transmitted_sparse_point_count": len(motion_sparse_rows),
        "transmitted_sparse_points_dominated_by_dense_same_rate": sum(
            bool(row["dominated_by_dense_same_rate"]) for row in motion_sparse_rows
        ),
        "all_transmitted_sparse_points_dominated_by_dense_same_rate": all(
            bool(row["dominated_by_dense_same_rate"]) for row in motion_sparse_rows
        ),
        "unmeasured_costs": [
            "audio side-information link",
            "reference face or keyframe",
            "modulation and channel coding",
            "protocol headers and synchronization",
        ],
        "source_provenance": source,
    }
    atomic_write_json(run_dir / "resolved_config.json", dict(settings.config))
    atomic_write_json(run_dir / "source_provenance.json", source)
    atomic_write_json(run_dir / "environment.json", _environment())
    atomic_write_json(
        run_dir / "run_metadata.json",
        {
            "report_fingerprint": fingerprint,
            "git_commit": _git_commit(),
            "model_selection_performed": False,
            "test_metrics_recomputed": False,
        },
    )
    atomic_write_json(run_dir / "accounting.json", {"rows": accounting_rows})
    _write_dict_csv(run_dir / "accounting.csv", accounting_rows)
    atomic_write_json(run_dir / "motion_rate_quality.json", {"rows": motion_rows})
    _write_dict_csv(run_dir / "motion_rate_quality.csv", motion_rows)
    atomic_write_json(run_dir / "video_rate_quality.json", {"rows": video_rows})
    _write_dict_csv(run_dir / "video_rate_quality.csv", video_rows)
    atomic_write_json(run_dir / "summary.json", summary)
    _write_plots(run_dir / "plots", motion_rows, video_rows)
    atomic_write_json(
        complete_path,
        {
            "report_fingerprint": fingerprint,
            "status": "complete",
            "motion_row_count": len(motion_rows),
            "video_row_count": len(video_rows),
        },
    )
    return run_dir, summary


def _motion_rate_quality(
    settings: CommunicationReportSettings,
    scorer_summary: Mapping[str, Any],
    gate_summary: Mapping[str, Any],
    gate_policy: GatePolicy,
) -> list[dict[str, Any]]:
    if scorer_summary.get("status") != "complete":
        raise ValueError("frozen scorer evaluation is not complete")
    if float(scorer_summary.get("maximum_dense_metric_difference", -1.0)) != 0.0:
        raise ValueError("scorer dense cross-check is not exact")
    aggregate = scorer_summary.get("aggregate")
    gate_groups = gate_summary.get("groups")
    if not isinstance(aggregate, list) or not isinstance(gate_groups, list):
        raise ValueError("missing frozen scorer or gate aggregates")
    gate_index = {(int(row["channel_uses"]), float(row["snr_db"])): row for row in gate_groups}
    prediction_by_snr: dict[float, Mapping[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for raw in aggregate:
        method = str(raw["method"])
        if method not in settings.methods:
            continue
        channel_uses = int(raw["channel_uses"])
        snr_db = float(raw["snr_db"])
        gate = gate_index[(channel_uses, snr_db)]
        transmit = gate_policy.should_transmit(channel_uses, snr_db)
        if bool(raw["gate_transmit"]) != transmit:
            raise ValueError("scorer and frozen gate decisions differ")
        if method == "dense_jscc":
            for metric, gate_key in (
                ("l1_mean", "gated_l1"),
                ("rmse_mean", "gated_rmse"),
                ("velocity_l1_mean", "gated_velocity_l1"),
                ("normalized_residual_mse_mean", "gated_normalized_residual_mse"),
            ):
                if abs(float(raw[metric]) - float(gate[gate_key])) > 1e-12:
                    raise ValueError("dense scorer aggregate differs from frozen gate")
        prediction_by_snr[snr_db] = gate
        cost = _cost_fields(
            settings,
            channel_uses=channel_uses,
            semantic_dimension_count=int(raw["k"]),
            transmit=transmit,
        )
        prediction_l1 = float(gate["prediction_l1"])
        row = {
            "scope": "frozen_test_aggregate",
            "method": method,
            "channel_uses": channel_uses,
            "snr_db": snr_db,
            "gate_transmit": transmit,
            "seed_count": int(raw["seed_count"]),
            **cost,
            **{metric: float(raw[metric]) for metric in _MOTION_METRICS},
            "l1_std": float(raw["l1_std"]),
            "rmse_std": float(raw["rmse_std"]),
            "velocity_l1_std": float(raw["velocity_l1_std"]),
            "l1_improvement_vs_prediction_percent": (
                (prediction_l1 - float(raw["l1_mean"])) / prediction_l1 * 100.0
            ),
            "dominated_by_dense_same_rate": False,
            "pareto_optimal_l1": False,
        }
        rows.append(row)
    expected = len(settings.methods) * len(gate_index)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} motion report rows, got {len(rows)}")
    for snr_db, gate in sorted(prediction_by_snr.items()):
        rows.append(
            {
                "scope": "frozen_test_aggregate",
                "method": "prediction_only",
                "channel_uses": 0,
                "snr_db": snr_db,
                "gate_transmit": False,
                "seed_count": 0,
                **_cost_fields(
                    settings,
                    channel_uses=0,
                    semantic_dimension_count=0,
                    transmit=False,
                ),
                "l1_mean": float(gate["prediction_l1"]),
                "rmse_mean": float(gate["prediction_rmse"]),
                "velocity_l1_mean": float(gate["prediction_velocity_l1"]),
                "normalized_residual_mse_mean": float(gate["prediction_normalized_residual_mse"]),
                "l1_std": 0.0,
                "rmse_std": 0.0,
                "velocity_l1_std": 0.0,
                "l1_improvement_vs_prediction_percent": 0.0,
                "dominated_by_dense_same_rate": False,
                "pareto_optimal_l1": False,
            }
        )
    dense_index = {
        (row["channel_uses"], row["snr_db"]): row for row in rows if row["method"] == "dense_jscc"
    }
    for row in rows:
        if row["method"] not in {"raw_magnitude", "learned_scorer"}:
            continue
        dense = dense_index[(row["channel_uses"], row["snr_db"])]
        row["dominated_by_dense_same_rate"] = dense["complex_symbols_per_clip"] == row[
            "complex_symbols_per_clip"
        ] and float(dense["l1_mean"]) < float(row["l1_mean"])
    _mark_pareto(rows, quality_key="l1_mean", output_key="pareto_optimal_l1")
    return sorted(
        rows,
        key=lambda row: (
            float(row["snr_db"]),
            int(row["complex_symbols_per_clip"]),
            str(row["method"]),
        ),
    )


def _video_rate_quality(
    settings: CommunicationReportSettings,
    video_summary: Mapping[str, Any],
    gate_policy: GatePolicy,
) -> list[dict[str, Any]]:
    if int(video_summary.get("failure_count", -1)) != 0:
        raise ValueError("E5 video reconstruction contains failures")
    groups = video_summary.get("groups")
    if not isinstance(groups, list):
        raise ValueError("E5 video summary has no groups")
    prediction = next(row for row in groups if row["family"] == "prediction_only")
    jscc_index = {
        (int(row["channel_uses"]), float(row["snr_db"])): row
        for row in groups
        if row["family"] == "jscc_awgn"
    }
    rows: list[dict[str, Any]] = []
    for channel_uses, snr_db in sorted(jscc_index):
        transmit = gate_policy.should_transmit(channel_uses, snr_db)
        source = jscc_index[(channel_uses, snr_db)] if transmit else prediction
        rows.append(
            {
                "scope": "frozen_test_dense_video_noise_seed_42",
                "method": "dense_jscc" if transmit else "prediction_only_gate_fallback",
                "channel_uses": channel_uses,
                "snr_db": snr_db,
                "gate_transmit": transmit,
                "noise_seed": None if not transmit else int(source["noise_seed"]),
                **_cost_fields(
                    settings,
                    channel_uses=channel_uses,
                    semantic_dimension_count=settings.motion_dimension,
                    transmit=transmit,
                ),
                "oracle_mouth_mae": float(source["oracle_mouth_mae"]),
                "oracle_mouth_nme": float(source["oracle_mouth_nme"]),
                "oracle_psnr_db": float(source["oracle_psnr_db"]),
                "oracle_ssim": float(source["oracle_ssim"]),
                "landmark_coverage": float(source["oracle_landmark_coverage"]),
                "pareto_optimal_mouth_nme": False,
            }
        )
    _mark_pareto(
        rows,
        quality_key="oracle_mouth_nme",
        output_key="pareto_optimal_mouth_nme",
    )
    return rows


def _accounting_rows(
    settings: CommunicationReportSettings,
    gate_policy: GatePolicy,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for channel_uses in sorted(gate_policy.thresholds_db):
        for semantic_dimension_count, representation in (
            (settings.motion_dimension, "dense_18d_input"),
            (2 * channel_uses, "hard_top_k_input"),
        ):
            rows.append(
                {
                    "representation": representation,
                    "channel_uses": channel_uses,
                    **_cost_fields(
                        settings,
                        channel_uses=channel_uses,
                        semantic_dimension_count=semantic_dimension_count,
                        transmit=True,
                    ),
                }
            )
    return rows


def _cost_fields(
    settings: CommunicationReportSettings,
    *,
    channel_uses: int,
    semantic_dimension_count: int,
    transmit: bool,
) -> dict[str, Any]:
    actual_channel_uses = channel_uses if transmit else 0
    complex_symbols = actual_channel_uses * settings.eligible_frame_count
    selected_values = semantic_dimension_count * settings.eligible_frame_count if transmit else 0
    return {
        "nominal_complex_channel_uses_per_eligible_frame": channel_uses,
        "actual_complex_channel_uses_per_eligible_frame": actual_channel_uses,
        "eligible_frames_per_clip": settings.eligible_frame_count,
        "complex_symbols_per_clip": complex_symbols,
        "real_channel_degrees_of_freedom_per_clip": 2 * complex_symbols,
        "complex_symbols_per_second_clip_average": (
            complex_symbols / settings.clip_duration_seconds
        ),
        "semantic_dimension_count_before_jscc": (semantic_dimension_count if transmit else 0),
        "semantic_values_before_jscc_per_clip": selected_values,
        "semantic_keep_ratio": (
            semantic_dimension_count / settings.motion_dimension if transmit else 0.0
        ),
        "digital_bitrate_bits_per_second": None,
        "explicit_selection_index_bits_per_clip": None,
    }


def _mark_pareto(
    rows: Sequence[dict[str, Any]],
    *,
    quality_key: str,
    output_key: str,
) -> None:
    for snr_db in sorted({float(row["snr_db"]) for row in rows}):
        members = [row for row in rows if float(row["snr_db"]) == snr_db]
        for candidate in members:
            candidate_rate = int(candidate["complex_symbols_per_clip"])
            candidate_quality = float(candidate[quality_key])
            dominated = any(
                int(other["complex_symbols_per_clip"]) <= candidate_rate
                and float(other[quality_key]) <= candidate_quality
                and (
                    int(other["complex_symbols_per_clip"]) < candidate_rate
                    or float(other[quality_key]) < candidate_quality
                )
                for other in members
                if other is not candidate
            )
            candidate[output_key] = not dominated


def _source_provenance(
    e5_run_dir: Path,
    gate_run_dir: Path,
    scorer_run_dir: Path,
    ablation_run_dir: Path,
) -> dict[str, Any]:
    e5_metadata = _read_json(e5_run_dir / "run_metadata.json")
    e5_fingerprint = str(e5_metadata.get("experiment_fingerprint", ""))
    e5_complete = _read_json(e5_run_dir / "evaluation_complete.json")
    if (
        not e5_fingerprint
        or e5_complete.get("status") != "complete"
        or e5_complete.get("experiment_fingerprint") != e5_fingerprint
    ):
        raise ValueError("E5 evaluation source is not complete")
    test_metrics_hash = file_sha256(e5_run_dir / "test_metrics.jsonl")
    gate_complete = _read_json(gate_run_dir / "complete.json")
    scorer_complete = _read_json(scorer_run_dir / "evaluation_complete.json")
    ablation_complete = _read_json(ablation_run_dir / "audit_complete.json")
    policy = GatePolicy.from_dict(_read_json(gate_run_dir / "policy.json"))
    if policy.experiment_fingerprint != e5_fingerprint:
        raise ValueError("gate policy does not belong to E5")
    if gate_complete.get("source_test_metrics_sha256") != test_metrics_hash:
        raise ValueError("gate source test hash differs from E5")
    if scorer_complete.get("source_test_metrics_sha256") != test_metrics_hash:
        raise ValueError("scorer source test hash differs from E5")
    if (
        gate_complete.get("status") != "complete"
        or scorer_complete.get("status") != "complete"
        or ablation_complete.get("status") != "complete"
    ):
        raise ValueError("one frozen source experiment is incomplete")
    return {
        "e5_experiment_fingerprint": e5_fingerprint,
        "e5_test_metrics_sha256": test_metrics_hash,
        "e5_evaluation_complete_sha256": file_sha256(e5_run_dir / "evaluation_complete.json"),
        "e5_video_summary_sha256": file_sha256(e5_run_dir / "video_reconstruction/summary.json"),
        "gate_policy_sha256": file_sha256(gate_run_dir / "policy.json"),
        "gate_test_summary_sha256": file_sha256(gate_run_dir / "test_summary.json"),
        "scorer_evaluation_summary_sha256": file_sha256(scorer_run_dir / "evaluation_summary.json"),
        "scorer_ablation_audit_summary_sha256": file_sha256(
            ablation_run_dir / "audit_summary.json"
        ),
    }


def _write_plots(
    path: Path,
    motion_rows: Sequence[Mapping[str, Any]],
    video_rows: Sequence[Mapping[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    path.mkdir(parents=True, exist_ok=True)
    methods = ("prediction_only", "dense_jscc", "raw_magnitude", "learned_scorer")
    for snr_db in sorted({float(row["snr_db"]) for row in motion_rows}):
        figure, axis = plt.subplots(figsize=(7.5, 4.8))
        for method in methods:
            members = [
                row
                for row in motion_rows
                if row["method"] == method and float(row["snr_db"]) == snr_db
            ]
            if not members:
                continue
            axis.plot(
                [row["complex_symbols_per_second_clip_average"] for row in members],
                [row["l1_mean"] for row in members],
                marker="o",
                label=method,
            )
        axis.set_title(f"Frozen test motion rate-quality, SNR={snr_db:g} dB")
        axis.set_xlabel("complex channel symbols / second")
        axis.set_ylabel("raw motion L1")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(path / f"motion_l1_snr_{snr_db:g}.png", dpi=160)
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for snr_db in sorted({float(row["snr_db"]) for row in video_rows}):
        members = [row for row in video_rows if float(row["snr_db"]) == snr_db]
        axes[0].plot(
            [row["complex_symbols_per_second_clip_average"] for row in members],
            [row["oracle_mouth_nme"] for row in members],
            marker="o",
            label=f"{snr_db:g} dB",
        )
        axes[1].plot(
            [row["complex_symbols_per_second_clip_average"] for row in members],
            [row["oracle_mouth_mae"] for row in members],
            marker="o",
            label=f"{snr_db:g} dB",
        )
    axes[0].set_ylabel("mouth NME vs lip-only oracle")
    axes[1].set_ylabel("mouth ROI MAE vs lip-only oracle")
    for axis in axes:
        axis.set_xlabel("complex channel symbols / second")
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path / "dense_video_rate_quality.png", dpi=160)
    plt.close(figure)
