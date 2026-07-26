"""Validate a completed E3 run and print its motion-level summary."""

from __future__ import annotations

import json

from av_semcom.models.predictor.cli import (
    build_predictor_parser,
    predictor_settings_from_args,
)
from av_semcom.models.predictor.evaluation import validate_audio_motion_run


def main() -> int:
    parser = build_predictor_parser("Validate audio-to-motion predictions and metrics.")
    args = parser.parse_args()
    if args.run_dir is None:
        parser.error("--run-dir is required")
    try:
        _, data_settings, settings, _ = predictor_settings_from_args(args)
        report = validate_audio_motion_run(args.run_dir, settings, data_settings)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report, indent=2))
    return 0 if report["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
