"""Train the Sionna complex-AWGN residual JSCC baseline."""

from __future__ import annotations

import json

from av_semcom.data.grid import read_manifest
from av_semcom.models.jscc.cli import build_jscc_parser, jscc_settings_from_args
from av_semcom.models.jscc.experiment import run_jscc_training
from av_semcom.models.predictor.data import select_predictor_samples


def main() -> int:
    parser = build_jscc_parser(__doc__ or "Train residual JSCC.")
    args = parser.parse_args()
    try:
        _, data, predictor, jscc = jscc_settings_from_args(args)
        samples = select_predictor_samples(read_manifest(data.manifest_path), data)
        run_dir, summary = run_jscc_training(
            jscc,
            predictor,
            samples,
            args.e3_run_dir,
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
                "model_count": len(summary["models"]),
                "channel_backend": summary["channel_backend"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
