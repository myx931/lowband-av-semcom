"""Train and evaluate the three-seed causal audio-to-motion baseline."""

from __future__ import annotations

import json

from av_semcom.data.grid import read_manifest
from av_semcom.models.predictor.cli import (
    build_predictor_parser,
    predictor_settings_from_args,
)
from av_semcom.models.predictor.data import select_predictor_samples
from av_semcom.models.predictor.experiment import run_audio_motion_experiment


def main() -> int:
    parser = build_predictor_parser("Train the causal audio-to-motion GRU baseline.")
    args = parser.parse_args()
    try:
        _, data_settings, settings, _ = predictor_settings_from_args(args)
        samples = select_predictor_samples(
            read_manifest(data_settings.manifest_path),
            data_settings,
        )
        run_directory, summary = run_audio_motion_experiment(
            settings,
            samples,
            run_directory=args.run_dir,
            resume=args.resume,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        json.dumps(
            {
                "run_directory": str(run_directory),
                "result_count": summary["result_count"],
                "e4_validation_gate_passed": summary["e4_validation_gate_passed"],
                "rq1_test_mean_baseline_improved": summary["rq1_test_mean_baseline_improved"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
