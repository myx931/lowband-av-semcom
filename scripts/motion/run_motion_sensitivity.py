"""Run the fixed E2 mouth-motion perturbation grid."""

from __future__ import annotations

import json

from av_semcom.models.motion.cli import build_motion_parser, motion_settings_from_args
from av_semcom.models.motion.experiment import run_motion_sensitivity


def main() -> int:
    parser = build_motion_parser("Run mouth-motion reconstruction sensitivity.")
    args = parser.parse_args()
    try:
        _, _, settings = motion_settings_from_args(args)
        run_directory, summary, failures = run_motion_sensitivity(settings)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        json.dumps(
            {
                "run_directory": str(run_directory),
                "result_count": summary["result_count"],
                "condition_count": summary["condition_count"],
                "failure_count": len(failures),
            },
            indent=2,
        )
    )
    return 0 if summary["result_count"] and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
