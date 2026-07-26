"""Extract pinned-backend mouth motion for a GRID manifest."""

from __future__ import annotations

import json

from av_semcom.data.grid import read_manifest
from av_semcom.models.motion.cli import build_motion_parser, motion_settings_from_args
from av_semcom.models.motion.pipeline import extract_motion_for_manifest


def main() -> int:
    parser = build_motion_parser("Extract LivePortrait mouth motion.")
    args = parser.parse_args()
    try:
        _, _, settings = motion_settings_from_args(args)
        samples = read_manifest(settings.data_settings.manifest_path)
        updated, failures, normalizer = extract_motion_for_manifest(
            settings,
            samples,
            overwrite=args.overwrite,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        json.dumps(
            {
                "sample_count": len(updated),
                "motion_count": sum(sample.motion_path is not None for sample in updated),
                "failure_count": len(failures),
                "stats_scope": normalizer.scope if normalizer else None,
            },
            indent=2,
        )
    )
    return 0 if updated and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
