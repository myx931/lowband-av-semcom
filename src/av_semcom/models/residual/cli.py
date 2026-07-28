"""Command-line parsing shared by residual experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from av_semcom.data.grid import GridSettings
from av_semcom.models.motion.config import MotionSettings
from av_semcom.models.predictor.config import AudioMotionSettings
from av_semcom.models.residual.config import ResidualSettings
from av_semcom.utils.config import Config, load_yaml_config


def build_residual_parser(description: str) -> argparse.ArgumentParser:
    """Build the common E4 CLI."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/residual_baseline_ten_speaker.yaml"),
    )
    parser.add_argument("--e3-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reconstruction-batch-size",
        type=_positive_int,
        help="Override LivePortrait reconstruction batch size for this run.",
    )
    parser.add_argument(
        "--metric-workers",
        type=_positive_int,
        help="Override CPU reconstruction metric workers for this run.",
    )
    return parser


def residual_settings_from_args(
    args: argparse.Namespace,
) -> tuple[Config, GridSettings, AudioMotionSettings, MotionSettings, ResidualSettings]:
    """Resolve data, frozen predictor, renderer, and residual settings."""

    config = load_yaml_config(args.config)
    data = GridSettings.from_config(config)
    predictor = AudioMotionSettings.from_config(config, data)
    motion = MotionSettings.from_config(config, data)
    residual = ResidualSettings.from_config(config)
    if args.reconstruction_batch_size is not None:
        motion = replace(
            motion,
            reconstruction_batch_size=args.reconstruction_batch_size,
        )
    if args.metric_workers is not None:
        residual = replace(residual, metric_workers=args.metric_workers)
    return config, data, predictor, motion, residual


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed
