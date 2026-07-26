"""Validate motion artifact schemas and manifest references."""

from __future__ import annotations

import json

from av_semcom.models.motion.cli import build_motion_parser, motion_settings_from_args
from av_semcom.models.motion.pipeline import validate_motion_manifest


def main() -> int:
    parser = build_motion_parser("Validate extracted GRID motion artifacts.")
    args = parser.parse_args()
    try:
        _, _, settings = motion_settings_from_args(args)
        report = validate_motion_manifest(settings)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.valid_count and report.error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
