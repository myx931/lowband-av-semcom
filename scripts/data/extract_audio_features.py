"""Extract frame-aligned log-Mel features for a GRID manifest."""

from __future__ import annotations

import json

from av_semcom.data.cli import build_data_parser, settings_from_args
from av_semcom.data.grid import read_manifest
from av_semcom.data.pipeline import extract_audio_for_manifest


def main() -> int:
    """Run the audio feature stage."""

    parser = build_data_parser("Extract GRID log-Mel audio features.")
    args = parser.parse_args()
    try:
        settings = settings_from_args(args)
        samples = read_manifest(settings.manifest_path)
        samples, failures = extract_audio_for_manifest(
            settings,
            samples,
            overwrite=args.overwrite,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps({"sample_count": len(samples), "failure_count": len(failures)}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
