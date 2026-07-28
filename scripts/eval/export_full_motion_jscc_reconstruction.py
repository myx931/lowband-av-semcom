"""Export frozen full-motion Sionna candidates for LivePortrait."""

from __future__ import annotations

import json

from av_semcom.models.full_motion.cli import (
    build_full_motion_parser,
    settings_from_args,
)
from av_semcom.models.full_motion.export import export_full_motion_candidates
from av_semcom.models.jscc.config import JSCCReconstructionSettings


def main() -> int:
    parser = build_full_motion_parser(
        __doc__ or "Export full-motion JSCC candidates.",
        evaluation=True,
    )
    args = parser.parse_args()
    try:
        config, predictor, jscc = settings_from_args(args)
        reconstruction = JSCCReconstructionSettings.from_config(config, jscc)
        result = export_full_motion_candidates(
            jscc,
            reconstruction,
            predictor.motion_stats_path,
            args.source_e5_run_dir,
            args.run_dir,
            resume=args.resume,
            formal=True,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
