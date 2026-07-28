"""Evaluate scorer ablations only on the reserved validation audit partition."""

from __future__ import annotations

import json

from av_semcom.models.selection.cli import (
    build_residual_scorer_ablation_parser,
    residual_scorer_ablation_settings_from_args,
)
from av_semcom.models.selection.scorer_ablation import (
    run_scorer_ablation_evaluation,
)


def main() -> int:
    parser = build_residual_scorer_ablation_parser(
        __doc__ or "Evaluate scorer ablation.",
        evaluation=True,
    )
    args = parser.parse_args()
    try:
        _, _, predictor, jscc, scorer, ablation = residual_scorer_ablation_settings_from_args(args)
        summary = run_scorer_ablation_evaluation(
            ablation,
            scorer,
            jscc,
            predictor,
            args.e5_run_dir,
            args.run_dir,
            resume=args.resume,
            formal=True,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        json.dumps(
            {
                "run_directory": str(args.run_dir.resolve()),
                "result_count": summary["result_count"],
                "evaluation_scope": summary["evaluation_scope"],
                "test_data_accessed": summary["test_data_accessed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
