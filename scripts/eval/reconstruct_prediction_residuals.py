"""Render and score fixed-budget E4 prediction-residual candidates."""

from __future__ import annotations

import json

from av_semcom.data.grid import read_manifest
from av_semcom.models.predictor.data import select_predictor_samples
from av_semcom.models.residual.cli import (
    build_residual_parser,
    residual_settings_from_args,
)
from av_semcom.models.residual.reconstruction import run_residual_reconstruction


def main() -> int:
    parser = build_residual_parser(__doc__ or "Reconstruct prediction residuals.")
    args = parser.parse_args()
    if args.run_dir is None:
        parser.error("--run-dir is required")
    try:
        _, data, _, motion, residual = residual_settings_from_args(args)
        samples = select_predictor_samples(read_manifest(data.manifest_path), data)
        summary, failures = run_residual_reconstruction(
            residual,
            motion,
            samples,
            args.e3_run_dir,
            args.run_dir,
            resume=args.resume,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        json.dumps(
            {
                "output_directory": str(args.run_dir / "reconstruction"),
                "result_count": summary["result_count"],
                "failure_count": len(failures),
            },
            indent=2,
        )
    )
    return 0 if summary["result_count"] and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
