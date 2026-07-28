"""Train the validation-only 2x2 residual scorer ablation."""

from __future__ import annotations

import json

from av_semcom.models.selection.cli import (
    build_residual_scorer_ablation_parser,
    residual_scorer_ablation_settings_from_args,
)
from av_semcom.models.selection.scorer_ablation import (
    run_scorer_ablation_training,
)


def main() -> int:
    parser = build_residual_scorer_ablation_parser(__doc__ or "Train scorer ablation.")
    args = parser.parse_args()
    try:
        _, _, predictor, jscc, scorer, ablation = residual_scorer_ablation_settings_from_args(args)
        run_dir, summary = run_scorer_ablation_training(
            ablation,
            scorer,
            jscc,
            predictor,
            args.e5_run_dir,
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
                "model_count": summary["model_count"],
                "test_data_accessed": summary["test_data_accessed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
