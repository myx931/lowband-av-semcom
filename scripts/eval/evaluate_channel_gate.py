"""Calibrate the E6 SNR safety gate on validation and evaluate frozen test rows."""

from __future__ import annotations

import json

from av_semcom.models.selection.cli import (
    build_channel_gate_parser,
    channel_gate_settings_from_args,
)
from av_semcom.models.selection.gate import run_channel_gate_experiment


def main() -> int:
    parser = build_channel_gate_parser(__doc__ or "Evaluate channel gate.")
    args = parser.parse_args()
    try:
        _, _, predictor, jscc, gate = channel_gate_settings_from_args(args)
        run_dir, summary = run_channel_gate_experiment(
            gate,
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
                "validation_only_policy": True,
                "test_result_count": summary["result_count"],
                "thresholds_db": summary["policy"]["thresholds_db"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
