"""Train the matched full-motion Sionna JSCC control."""

from __future__ import annotations

import json

from av_semcom.models.full_motion.cli import (
    build_full_motion_parser,
    settings_from_args,
)
from av_semcom.models.full_motion.experiment import run_full_motion_training


def main() -> int:
    parser = build_full_motion_parser(__doc__ or "Train full-motion JSCC.")
    args = parser.parse_args()
    try:
        _, predictor, jscc = settings_from_args(args)
        run_dir, summary = run_full_motion_training(
            jscc,
            predictor.motion_stats_path,
            args.source_e5_run_dir,
            run_directory=args.run_dir,
            resume=args.resume,
            formal=True,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        json.dumps(
            {
                "run_directory": str(run_dir),
                "model_count": len(summary["models"]),
                "representation": summary["representation"],
                "bitrate_claimed": summary["bitrate_claimed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
