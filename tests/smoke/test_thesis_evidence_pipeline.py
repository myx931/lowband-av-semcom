from __future__ import annotations

import json
from pathlib import Path

import pytest

from av_semcom.analysis.thesis_evidence import (
    ThesisEvidenceSettings,
    ThesisSourceRuns,
    run_thesis_evidence,
)
from av_semcom.data.preprocessing import atomic_write_json

pytestmark = pytest.mark.smoke


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


def _video_rows(sample_id: str, *, residual: bool) -> list[dict[str, object]]:
    rows = []
    for channel_uses in (1, 2, 3, 4):
        for snr_db in (-5.0, 0.0, 5.0, 10.0):
            base = 0.02 + channel_uses / 1000
            adjustment = 0.0 if residual else 0.002
            rows.append(
                {
                    "sample_id": sample_id,
                    "family": "jscc_awgn",
                    "channel_uses": channel_uses,
                    "snr_db": snr_db,
                    "noise_seed": 42,
                    "oracle_mouth_mae": (base + adjustment) * 100,
                    "oracle_mouth_nme": base + adjustment,
                }
            )
    return rows


def _build_sources(tmp_path: Path) -> ThesisSourceRuns:
    roots = {
        name: tmp_path / name
        for name in (
            "e3",
            "e4",
            "residual",
            "gate",
            "scorer",
            "ablation",
            "communication",
            "full",
            "comparison",
        )
    }
    for root in roots.values():
        root.mkdir(parents=True)

    e3_groups = []
    e3_video_groups = []
    for method, l1, nme in (
        ("train_mean", 0.3, 0.03),
        ("zero_motion", 0.4, 0.04),
        ("audio_gru", 0.2, 0.02),
        ("oracle_persistence", 0.1, 0.01),
    ):
        seed = 42 if method == "audio_gru" else None
        e3_groups.append(
            {
                "method": method,
                "seed": seed,
                "split": "test",
                "sample_count": 2,
                "l1": l1,
                "rmse": l1 * 2,
                "velocity_l1": l1 / 2,
            }
        )
        e3_video_groups.append(
            {
                "method": method,
                "seed": seed,
                "split": "test",
                "oracle_mouth_mae": nme * 100,
                "oracle_mouth_nme": nme,
                "oracle_psnr_db": 40.0,
                "oracle_ssim": 0.98,
            }
        )
    e3_groups.append(
        {
            "method": "audio_gru",
            "seed": 42,
            "split": "validation",
            "sample_count": 2,
            "l1": 0.2,
            "rmse": 0.4,
            "velocity_l1": 0.1,
        }
    )
    atomic_write_json(roots["e3"] / "summary.json", {"groups": e3_groups})
    atomic_write_json(
        roots["e3"] / "reconstruction/summary.json",
        {"groups": e3_video_groups},
    )

    e4_groups = []
    for k in (0, 1, 2, 4, 6, 9, 12, 18):
        condition = (
            "prediction_only"
            if k == 0
            else "full_residual_oracle"
            if k == 18
            else "magnitude_top_k"
        )
        e4_groups.append(
            {
                "condition": condition,
                "selection_space": "none" if k in {0, 18} else "raw",
                "split": "test",
                "k": k,
                "l1": (18 - k) / 100.0,
                "rmse": (18 - k) / 50.0,
                "velocity_l1": (18 - k) / 200.0,
                "raw_energy_retained_fraction": k / 18.0,
            }
        )
    atomic_write_json(roots["e4"] / "summary.json", {"groups": e4_groups})
    atomic_write_json(
        roots["e4"] / "reconstruction/summary.json",
        {
            "groups": [
                {
                    "condition": "prediction_only",
                    "selection_space": "none",
                    "split": "test",
                    "k": 0,
                    "oracle_mouth_mae": 5.0,
                    "oracle_mouth_nme": 0.02,
                },
                {
                    "condition": "full_residual_oracle",
                    "selection_space": "raw",
                    "split": "test",
                    "k": 18,
                    "oracle_mouth_mae": 0.0,
                    "oracle_mouth_nme": 0.0,
                },
            ]
        },
    )
    atomic_write_json(roots["e4"] / "analysis_complete.json", {"status": "complete"})

    gate_groups = []
    motion_groups = []
    comparison_video_groups = []
    residual_video_groups = [
        {
            "family": "prediction_only",
            "oracle_mouth_mae": 3.0,
            "oracle_mouth_nme": 0.03,
        }
    ]
    motion_pairs = []
    scorer_aggregate = []
    for channel_uses in (1, 2, 3, 4):
        for snr_db in (-5.0, 0.0, 5.0, 10.0):
            residual_l1 = 0.2 - snr_db / 1000 + channel_uses / 10000
            full_l1 = residual_l1 + 0.01
            transmit = snr_db >= 0
            gate_groups.append(
                {
                    "channel_uses": channel_uses,
                    "snr_db": snr_db,
                    "decision": "send_jscc" if transmit else "prediction_only",
                    "prediction_l1": 0.25,
                    "always_send_l1": residual_l1,
                    "gated_l1": residual_l1 if transmit else 0.25,
                    "always_send_relative_l1_improvement_vs_prediction": (1.0 - residual_l1 / 0.25),
                    "gated_relative_l1_improvement_vs_prediction": (
                        1.0 - (residual_l1 if transmit else 0.25) / 0.25
                    ),
                }
            )
            motion_groups.append(
                {
                    "channel_uses": channel_uses,
                    "snr_db": snr_db,
                    "complex_symbols_per_clip": channel_uses * 74,
                    "complex_symbols_per_second": channel_uses * 74 / 3,
                    "residual_l1": residual_l1,
                    "full_motion_l1": full_l1,
                }
            )
            comparison_video_groups.append(
                {
                    "channel_uses": channel_uses,
                    "snr_db": snr_db,
                    "residual_oracle_mouth_mae": 2.0,
                    "full_motion_oracle_mouth_mae": 2.2,
                    "residual_oracle_mouth_nme": 0.02,
                    "full_motion_oracle_mouth_nme": 0.022,
                }
            )
            residual_video_groups.append(
                {
                    "family": "jscc_awgn",
                    "channel_uses": channel_uses,
                    "snr_db": snr_db,
                }
            )
            for sample_index, sample_id in enumerate(("s1_a", "s1_b")):
                for noise_seed in (42, 43):
                    advantage = 0.01 + sample_index / 1000
                    motion_pairs.append(
                        {
                            "sample_id": sample_id,
                            "channel_uses": channel_uses,
                            "snr_db": snr_db,
                            "noise_seed": noise_seed,
                            "residual_advantage_l1": advantage,
                            "residual_advantage_rmse": advantage * 2,
                            "residual_advantage_velocity_l1": advantage / 2,
                        }
                    )
        for snr_db in (0.0, 5.0, 10.0):
            for method, value in (
                ("dense_jscc", 0.1),
                ("raw_magnitude", 0.12),
                ("learned_scorer", 0.11),
            ):
                scorer_aggregate.append(
                    {
                        "channel_uses": channel_uses,
                        "snr_db": snr_db,
                        "method": method,
                        "l1_mean": value,
                    }
                )

    atomic_write_json(roots["gate"] / "test_summary.json", {"groups": gate_groups})
    atomic_write_json(roots["gate"] / "complete.json", {"status": "complete"})
    atomic_write_json(
        roots["scorer"] / "evaluation_summary.json",
        {"aggregate": scorer_aggregate},
    )
    atomic_write_json(
        roots["scorer"] / "evaluation_complete.json",
        {"status": "complete"},
    )
    atomic_write_json(
        roots["ablation"] / "audit_summary.json",
        {
            "status": "complete",
            "evaluation_scope": "validation_only",
            "audit_sample_count": 2,
            "audit_speakers": ["s1"],
            "test_data_accessed": False,
        },
    )
    atomic_write_json(roots["ablation"] / "audit_complete.json", {"status": "complete"})
    atomic_write_json(
        roots["communication"] / "summary.json",
        {
            "status": "complete",
            "digital_bitrate_defined": False,
            "rate_unit": "complex_channel_symbols",
            "eligible_frame_count": 74,
            "motion_dimension": 18,
            "all_transmitted_sparse_points_dominated_by_dense_same_rate": True,
            "unmeasured_costs": ["audio side-information link"],
        },
    )
    atomic_write_json(roots["communication"] / "complete.json", {"status": "complete"})

    atomic_write_json(
        roots["residual"] / "video_reconstruction/summary.json",
        {"groups": residual_video_groups},
    )
    atomic_write_json(roots["residual"] / "evaluation_complete.json", {"status": "complete"})
    atomic_write_json(
        roots["residual"] / "video_reconstruction/complete.json",
        {"status": "complete"},
    )
    atomic_write_json(roots["full"] / "evaluation_complete.json", {"status": "complete"})
    atomic_write_json(
        roots["full"] / "video_reconstruction/complete.json",
        {"status": "complete"},
    )
    for sample_id in ("s1_a", "s1_b"):
        atomic_write_json(
            roots["residual"] / f"video_reconstruction/samples/{sample_id}.json",
            {"rows": _video_rows(sample_id, residual=True)},
        )
        atomic_write_json(
            roots["full"] / f"video_reconstruction/samples/{sample_id}.json",
            {"rows": _video_rows(sample_id, residual=False)},
        )
        (roots["residual"] / f"video_reconstruction/media/test/{sample_id}").mkdir(parents=True)
        (roots["full"] / f"video_reconstruction/media/test/{sample_id}").mkdir(parents=True)

    atomic_write_json(
        roots["comparison"] / "motion_summary.json",
        {"rows": motion_groups},
    )
    atomic_write_json(
        roots["comparison"] / "video_summary.json",
        {"rows": comparison_video_groups},
    )
    _write_jsonl(roots["comparison"] / "motion_pairs.jsonl", motion_pairs)
    atomic_write_json(roots["comparison"] / "complete.json", {"status": "complete"})

    return ThesisSourceRuns(
        e3=roots["e3"],
        e4=roots["e4"],
        residual_jscc=roots["residual"],
        gate=roots["gate"],
        scorer=roots["scorer"],
        scorer_ablation=roots["ablation"],
        communication=roots["communication"],
        full_motion=roots["full"],
        comparison=roots["comparison"],
    )


def test_thesis_evidence_pack_builds_and_resumes(tmp_path: Path) -> None:
    settings = ThesisEvidenceSettings.from_config(
        {
            "thesis_evidence": {
                "output_dir": str(tmp_path / "evidence"),
                "bootstrap_seed": 42,
                "bootstrap_resamples": 200,
                "confidence_level": 0.95,
                "expected_test_sample_count": 2,
                "qualitative_positions": [0, 1],
                "figure_dpi": 80,
            }
        }
    )
    sources = _build_sources(tmp_path)
    run_dir = tmp_path / "formal"

    output, summary = run_thesis_evidence(settings, sources, run_directory=run_dir)
    complete_before = (output / "complete.json").stat().st_mtime_ns
    resumed, resumed_summary = run_thesis_evidence(
        settings,
        sources,
        run_directory=run_dir,
        resume=True,
    )

    assert resumed == output
    assert resumed_summary == summary
    assert (output / "complete.json").stat().st_mtime_ns == complete_before
    assert summary["main_table_row_count"] == 16
    assert summary["motion_bootstrap_row_count"] == 48
    assert summary["video_bootstrap_row_count"] == 32
    assert summary["figure_count"] == 7
    assert summary["multiplicity_adjusted"] is False
    assert (output / "methodological_checks.json").is_file()
    assert len(list((output / "figures").glob("*.png"))) == 7
