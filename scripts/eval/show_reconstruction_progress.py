"""Show progress for a resumable audio-to-motion reconstruction run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from av_semcom.utils.run_progress import (
    format_duration,
    inspect_reconstruction_progress,
    wait_for_next_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--watch",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="refresh repeatedly at this interval; zero prints once",
    )
    args = parser.parse_args()
    if args.watch < 0:
        parser.error("--watch must be non-negative")

    while True:
        progress = inspect_reconstruction_progress(args.run_dir)
        if args.as_json:
            print(json.dumps(progress.to_dict(), ensure_ascii=False, indent=2), flush=True)
        else:
            rate = (
                f"{progress.samples_per_minute:.2f} samples/min"
                if progress.samples_per_minute is not None
                else "rate unknown"
            )
            print(
                f"[{progress.status}] {progress.completed_samples}/"
                f"{progress.total_samples} ({progress.percent_complete:.1f}%) | "
                f"{rate} | ETA {format_duration(progress.eta_seconds)} | "
                f"failures {progress.failed_samples}",
                flush=True,
            )
            if progress.gpu_status:
                print(f"GPU: {progress.gpu_status}", flush=True)
            if progress.summary_available:
                print(f"Summary: {args.run_dir / 'reconstruction' / 'summary.json'}")
        if not args.watch or progress.status == "complete":
            return 0
        wait_for_next_snapshot(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
