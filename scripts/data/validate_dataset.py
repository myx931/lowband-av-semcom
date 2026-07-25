"""Validate a GRID manifest and write a machine-readable report."""

from __future__ import annotations

import json

from av_semcom.data.cli import build_data_parser, settings_from_args
from av_semcom.data.grid import read_manifest
from av_semcom.data.preprocessing import atomic_write_json
from av_semcom.data.validation import validate_samples


def main() -> int:
    """Validate raw or processed manifest records."""

    parser = build_data_parser("Validate GRID paths, records, and split isolation.")
    parser.add_argument(
        "--require-processed",
        action="store_true",
        help="Require audio, landmark, and face-crop artifacts.",
    )
    args = parser.parse_args()
    try:
        settings = settings_from_args(args)
        samples = read_manifest(settings.manifest_path)
        report = validate_samples(
            samples,
            settings.data_root,
            require_processed=args.require_processed,
        )
        report_path = settings.manifest_path.with_name("validation_report.json")
        atomic_write_json(report_path, report.to_dict())
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.error_count == 0 and report.sample_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
