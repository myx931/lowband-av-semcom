"""Render and score frozen E5 JSCC candidates with LivePortrait."""

from __future__ import annotations

import json

from av_semcom.data.grid import read_manifest
from av_semcom.models.jscc.cli import (
    build_jscc_reconstruction_parser,
    jscc_reconstruction_settings_from_args,
)
from av_semcom.models.jscc.reconstruction import run_jscc_reconstruction
from av_semcom.models.predictor.data import select_predictor_samples


def main() -> int:
    parser = build_jscc_reconstruction_parser(
        __doc__ or "Reconstruct JSCC candidates.",
        render=True,
    )
    args = parser.parse_args()
    try:
        _, data, _, motion, jscc, reconstruction = jscc_reconstruction_settings_from_args(args)
        samples = select_predictor_samples(read_manifest(data.manifest_path), data)
        summary, failures = run_jscc_reconstruction(
            jscc,
            reconstruction,
            motion,
            samples,
            args.run_dir,
            resume=args.resume,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        json.dumps(
            {
                "output_directory": str(args.run_dir / "video_reconstruction"),
                "result_count": summary["result_count"],
                "failure_count": len(failures),
            },
            indent=2,
        )
    )
    return 0 if summary["result_count"] and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
