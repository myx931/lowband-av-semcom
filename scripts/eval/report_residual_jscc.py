"""Create multi-seed tables and SNR curves from completed E5 test metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from av_semcom.models.jscc.config import JSCCSettings
from av_semcom.models.jscc.experiment import write_jscc_report
from av_semcom.utils.config import load_yaml_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/residual_jscc_ten_speaker.yaml"),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        settings = JSCCSettings.from_config(load_yaml_config(args.config))
        summary = write_jscc_report(settings, args.run_dir)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        json.dumps(
            {
                "run_directory": str(args.run_dir.resolve()),
                "source_metrics_sha256": summary["source_metrics_sha256"],
                "aggregate_count": len(summary["seed_aggregate"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
