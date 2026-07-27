"""Extract bounded GRID JPG sequences from official MPG containers."""

from __future__ import annotations

import json

from av_semcom.data.cli import build_data_parser, settings_from_args
from av_semcom.data.video_frames import extract_grid_frame_sequences


def main() -> int:
    """Extract configured GRID frame sequences."""

    parser = build_data_parser("Extract GRID JPG sequences from MPG containers.")
    args = parser.parse_args()
    try:
        settings = settings_from_args(args)
        sample_count, processed, failures = extract_grid_frame_sequences(
            settings,
            overwrite=args.overwrite,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        json.dumps(
            {
                "sample_count": sample_count,
                "processed": processed,
                "failure_count": len(failures),
            },
            indent=2,
        )
    )
    return 0 if sample_count and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
