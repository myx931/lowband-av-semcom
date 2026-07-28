"""Train E6 hard Top-K residual scorers with frozen E5 JSCC weights."""

from __future__ import annotations

import json

from av_semcom.models.selection.cli import (
    build_residual_scorer_parser,
    residual_scorer_settings_from_args,
)
from av_semcom.models.selection.scorer_experiment import run_scorer_training


def main() -> int:
    parser = build_residual_scorer_parser(__doc__ or "Train residual scorer.")
    args = parser.parse_args()
    try:
        _, _, predictor, jscc, scorer = residual_scorer_settings_from_args(args)
        run_dir, summary = run_scorer_training(
            scorer,
            jscc,
            predictor,
            args.e5_run_dir,
            args.gate_run_dir,
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
                "jscc_weights_frozen": summary["jscc_weights_frozen"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
