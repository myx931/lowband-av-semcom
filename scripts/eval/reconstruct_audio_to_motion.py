"""Run full LivePortrait reconstruction evaluation for an E3 prediction run."""

from __future__ import annotations

import json

from av_semcom.data.grid import read_manifest
from av_semcom.models.predictor.cli import (
    build_predictor_parser,
    predictor_settings_from_args,
)
from av_semcom.models.predictor.data import select_predictor_samples
from av_semcom.models.predictor.reconstruction import run_prediction_reconstruction


def main() -> int:
    parser = build_predictor_parser("Reconstruct and score audio-to-motion predictions.")
    args = parser.parse_args()
    if args.run_dir is None:
        parser.error("--run-dir is required")
    try:
        _, data_settings, settings, motion_settings = predictor_settings_from_args(args)
        samples = select_predictor_samples(
            read_manifest(data_settings.manifest_path),
            data_settings,
        )
        summary, failures = run_prediction_reconstruction(
            settings,
            motion_settings,
            samples,
            args.run_dir,
            resume=args.resume,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        json.dumps(
            {
                "result_count": summary["result_count"],
                "failure_count": len(failures),
                "output_directory": str(args.run_dir / "reconstruction"),
            },
            indent=2,
        )
    )
    return 0 if summary["result_count"] and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
