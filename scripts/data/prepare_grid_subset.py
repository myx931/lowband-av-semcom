"""Discover paired GRID samples and write a portable manifest."""

from __future__ import annotations

import json

from av_semcom.data.cli import build_data_parser, settings_from_args
from av_semcom.data.pipeline import prepare_grid_subset


def main() -> int:
    """Prepare a configured GRID subset."""

    parser = build_data_parser("Prepare a small GRID JSONL manifest.")
    args = parser.parse_args()
    try:
        settings = settings_from_args(args)
        samples, failures, processed = prepare_grid_subset(
            settings,
            overwrite=args.overwrite,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        json.dumps(
            {
                "manifest": str(settings.manifest_path),
                "sample_count": len(samples),
                "failure_count": len(failures),
                "processed": processed,
            },
            indent=2,
        )
    )
    return 0 if samples and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
