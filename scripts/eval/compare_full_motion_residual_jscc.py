"""Compare frozen full-motion and residual JSCC at matched symbol budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from av_semcom.models.full_motion.comparison import run_matched_comparison
from av_semcom.models.full_motion.config import comparison_output_root
from av_semcom.utils.config import load_yaml_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/residual_jscc_ten_speaker.yaml"),
    )
    parser.add_argument("--residual-run-dir", type=Path, required=True)
    parser.add_argument("--full-motion-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        if args.resume and args.run_dir is None:
            raise ValueError("--resume requires --run-dir")
        output, summary = run_matched_comparison(
            comparison_output_root(load_yaml_config(args.config)),
            args.residual_run_dir,
            args.full_motion_run_dir,
            run_directory=args.run_dir,
            resume=args.resume,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        json.dumps(
            {
                "run_directory": str(output),
                "motion_pair_count": summary["motion_pair_count"],
                "motion_group_count": summary["motion_group_count"],
                "video_group_count": summary["video_group_count"],
                "digital_bitrate_defined": summary["digital_bitrate_defined"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
