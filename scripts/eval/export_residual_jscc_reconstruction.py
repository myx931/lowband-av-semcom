"""Export frozen Sionna motion candidates for LivePortrait reconstruction."""

from __future__ import annotations

import json

from av_semcom.models.jscc.cli import (
    build_jscc_reconstruction_parser,
    jscc_reconstruction_settings_from_args,
)
from av_semcom.models.jscc.export import export_jscc_reconstruction_candidates


def main() -> int:
    parser = build_jscc_reconstruction_parser(__doc__ or "Export JSCC candidates.")
    args = parser.parse_args()
    try:
        _, _, predictor, _, jscc, reconstruction = jscc_reconstruction_settings_from_args(args)
        result = export_jscc_reconstruction_candidates(
            jscc,
            reconstruction,
            predictor,
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
