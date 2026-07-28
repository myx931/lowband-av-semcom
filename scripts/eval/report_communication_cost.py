"""Report communication cost and rate-quality from frozen E5/E6 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from av_semcom.analysis.communication_report import (
    CommunicationReportSettings,
    run_communication_report,
)
from av_semcom.utils.config import load_yaml_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/residual_jscc_ten_speaker.yaml"),
    )
    parser.add_argument("--e5-run-dir", type=Path, required=True)
    parser.add_argument("--gate-run-dir", type=Path, required=True)
    parser.add_argument("--scorer-run-dir", type=Path, required=True)
    parser.add_argument("--ablation-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        if args.resume and args.run_dir is None:
            raise ValueError("--resume requires --run-dir")
        settings = CommunicationReportSettings.from_config(load_yaml_config(args.config))
        run_dir, summary = run_communication_report(
            settings,
            args.e5_run_dir,
            args.gate_run_dir,
            args.scorer_run_dir,
            args.ablation_run_dir,
            run_directory=args.run_dir,
            resume=args.resume,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        json.dumps(
            {
                "run_directory": str(run_dir),
                "motion_row_count": summary["motion_row_count"],
                "video_row_count": summary["video_row_count"],
                "digital_bitrate_defined": summary["digital_bitrate_defined"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
