"""Render and score frozen full-motion JSCC candidates with LivePortrait."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from av_semcom.data.grid import GridSettings, read_manifest
from av_semcom.models.full_motion.config import full_motion_jscc_settings
from av_semcom.models.jscc.config import JSCCReconstructionSettings
from av_semcom.models.jscc.reconstruction import run_jscc_reconstruction
from av_semcom.models.motion.config import MotionSettings
from av_semcom.models.predictor.data import select_predictor_samples
from av_semcom.utils.config import load_yaml_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/residual_jscc_ten_speaker.yaml"),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--reconstruction-batch-size", type=int)
    parser.add_argument("--metric-workers", type=int)
    args = parser.parse_args()
    try:
        config = load_yaml_config(args.config)
        data = GridSettings.from_config(config)
        motion = MotionSettings.from_config(config, data)
        jscc = full_motion_jscc_settings(config)
        reconstruction = JSCCReconstructionSettings.from_config(config, jscc)
        if args.device:
            motion = replace(motion, device=str(args.device))
        if args.reconstruction_batch_size is not None:
            if args.reconstruction_batch_size < 1:
                raise ValueError("reconstruction batch size must be positive")
            motion = replace(
                motion,
                reconstruction_batch_size=args.reconstruction_batch_size,
            )
        if args.metric_workers is not None:
            if args.metric_workers < 1:
                raise ValueError("metric workers must be positive")
            reconstruction = replace(
                reconstruction,
                metric_workers=args.metric_workers,
            )
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
