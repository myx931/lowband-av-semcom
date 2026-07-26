"""Run the full, mouth-only, and frozen reconstruction baselines."""

from __future__ import annotations

import json

from av_semcom.models.motion.cli import build_motion_parser, motion_settings_from_args
from av_semcom.models.motion.experiment import run_motion_sensitivity
from av_semcom.models.motion.perturbations import PerturbationCondition


def main() -> int:
    parser = build_motion_parser("Run LivePortrait reconstruction baselines.")
    args = parser.parse_args()
    try:
        _, _, settings = motion_settings_from_args(args)
        run_directory, summary, failures = run_motion_sensitivity(
            settings,
            conditions=(
                PerturbationCondition("lip_only", "identity"),
                PerturbationCondition("frozen", "frozen"),
            ),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        json.dumps(
            {
                "run_directory": str(run_directory),
                "result_count": summary["result_count"],
                "failure_count": len(failures),
            },
            indent=2,
        )
    )
    return 0 if summary["result_count"] and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
