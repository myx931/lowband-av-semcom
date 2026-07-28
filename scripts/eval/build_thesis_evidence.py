"""Build the E8 thesis evidence pack from frozen E3-E7 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from av_semcom.analysis.thesis_evidence import (
    ThesisEvidenceSettings,
    ThesisSourceRuns,
    run_thesis_evidence,
)
from av_semcom.utils.config import load_yaml_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/residual_jscc_ten_speaker.yaml"),
    )
    parser.add_argument("--e3-run-dir", type=Path, required=True)
    parser.add_argument("--e4-run-dir", type=Path, required=True)
    parser.add_argument("--residual-jscc-run-dir", type=Path, required=True)
    parser.add_argument("--gate-run-dir", type=Path, required=True)
    parser.add_argument("--scorer-run-dir", type=Path, required=True)
    parser.add_argument("--scorer-ablation-run-dir", type=Path, required=True)
    parser.add_argument("--communication-run-dir", type=Path, required=True)
    parser.add_argument("--full-motion-run-dir", type=Path, required=True)
    parser.add_argument("--comparison-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        if args.resume and args.run_dir is None:
            raise ValueError("--resume requires --run-dir")
        settings = ThesisEvidenceSettings.from_config(load_yaml_config(args.config))
        sources = ThesisSourceRuns(
            e3=args.e3_run_dir,
            e4=args.e4_run_dir,
            residual_jscc=args.residual_jscc_run_dir,
            gate=args.gate_run_dir,
            scorer=args.scorer_run_dir,
            scorer_ablation=args.scorer_ablation_run_dir,
            communication=args.communication_run_dir,
            full_motion=args.full_motion_run_dir,
            comparison=args.comparison_run_dir,
        )
        run_dir, summary = run_thesis_evidence(
            settings,
            sources,
            run_directory=args.run_dir,
            resume=args.resume,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        json.dumps(
            {
                "run_directory": str(run_dir),
                "main_table_row_count": summary["main_table_row_count"],
                "motion_bootstrap_row_count": summary["motion_bootstrap_row_count"],
                "video_bootstrap_row_count": summary["video_bootstrap_row_count"],
                "figure_count": summary["figure_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
