"""Evaluate frozen residual JSCC checkpoints on the held-out test speaker."""

from __future__ import annotations

import json

from av_semcom.data.grid import read_manifest
from av_semcom.models.jscc.cli import build_jscc_parser, jscc_settings_from_args
from av_semcom.models.jscc.experiment import run_jscc_evaluation
from av_semcom.models.predictor.data import select_predictor_samples


def main() -> int:
    parser = build_jscc_parser(__doc__ or "Evaluate residual JSCC.", evaluation=True)
    args = parser.parse_args()
    try:
        _, data, predictor, jscc = jscc_settings_from_args(args)
        samples = select_predictor_samples(read_manifest(data.manifest_path), data)
        summary = run_jscc_evaluation(
            jscc,
            predictor,
            samples,
            args.e3_run_dir,
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
                "channel_backend": summary["channel_backend"],
                "bitrate_claimed": summary["bitrate_claimed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
