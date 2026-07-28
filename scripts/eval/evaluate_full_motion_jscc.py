"""Evaluate the frozen full-motion JSCC models on the E5 test identities."""

from __future__ import annotations

import json

from av_semcom.models.full_motion.cli import (
    build_full_motion_parser,
    settings_from_args,
)
from av_semcom.models.full_motion.experiment import run_full_motion_evaluation


def main() -> int:
    parser = build_full_motion_parser(
        __doc__ or "Evaluate full-motion JSCC.",
        evaluation=True,
    )
    args = parser.parse_args()
    try:
        _, predictor, jscc = settings_from_args(args)
        summary = run_full_motion_evaluation(
            jscc,
            predictor.motion_stats_path,
            args.source_e5_run_dir,
            args.run_dir,
            resume=args.resume,
            formal=True,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        json.dumps(
            {
                "run_directory": str(args.run_dir.resolve()),
                "result_count": summary["result_count"],
                "representation": summary["representation"],
                "bitrate_claimed": summary["bitrate_claimed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
