"""Analyze fixed-seed audio-to-motion prediction residuals in motion space."""

from __future__ import annotations

import json

from av_semcom.data.grid import read_manifest
from av_semcom.models.predictor.data import select_predictor_samples
from av_semcom.models.residual.cli import (
    build_residual_parser,
    residual_settings_from_args,
)
from av_semcom.models.residual.experiment import run_residual_analysis


def main() -> int:
    parser = build_residual_parser(__doc__ or "Analyze prediction residuals.")
    args = parser.parse_args()
    try:
        _, data, predictor, _, residual = residual_settings_from_args(args)
        samples = select_predictor_samples(read_manifest(data.manifest_path), data)
        run_dir, summary = run_residual_analysis(
            residual,
            predictor,
            samples,
            args.e3_run_dir,
            run_directory=args.run_dir,
            resume=args.resume,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        json.dumps(
            {
                "output_directory": str(run_dir),
                "sample_count": summary["sample_count"],
                "selection_result_count": summary["selection_result_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
