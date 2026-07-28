"""Command-line configuration shared by E5 residual JSCC commands."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from av_semcom.data.grid import GridSettings
from av_semcom.models.jscc.config import (
    JSCCReconstructionSettings,
    JSCCSettings,
)
from av_semcom.models.motion.config import MotionSettings
from av_semcom.models.predictor.config import AudioMotionSettings
from av_semcom.utils.config import Config, load_yaml_config


def build_jscc_parser(
    description: str,
    *,
    evaluation: bool = False,
) -> argparse.ArgumentParser:
    """Build a strict E5 CLI parser."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/residual_jscc_ten_speaker.yaml"),
    )
    parser.add_argument("--e3-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=evaluation)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", help="Override jscc_training.device, for example cuda:0.")
    return parser


def jscc_settings_from_args(
    args: argparse.Namespace,
) -> tuple[Config, GridSettings, AudioMotionSettings, JSCCSettings]:
    """Resolve data, frozen predictor, and JSCC settings."""

    config = load_yaml_config(args.config)
    data = GridSettings.from_config(config)
    predictor = AudioMotionSettings.from_config(config, data)
    jscc = JSCCSettings.from_config(config)
    if args.device:
        jscc = replace(jscc, device=str(args.device))
    return config, data, predictor, jscc


def build_jscc_reconstruction_parser(
    description: str,
    *,
    render: bool = False,
) -> argparse.ArgumentParser:
    """Build the two-stage Sionna-export/LivePortrait-render parser."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/residual_jscc_ten_speaker.yaml"),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device")
    if render:
        parser.add_argument("--reconstruction-batch-size", type=_positive_int)
        parser.add_argument("--metric-workers", type=_positive_int)
    return parser


def jscc_reconstruction_settings_from_args(
    args: argparse.Namespace,
) -> tuple[
    Config,
    GridSettings,
    AudioMotionSettings,
    MotionSettings,
    JSCCSettings,
    JSCCReconstructionSettings,
]:
    """Resolve both environments' shared reconstruction protocol."""

    config = load_yaml_config(args.config)
    data = GridSettings.from_config(config)
    predictor = AudioMotionSettings.from_config(config, data)
    motion = MotionSettings.from_config(config, data)
    jscc = JSCCSettings.from_config(config)
    reconstruction = JSCCReconstructionSettings.from_config(config, jscc)
    if args.device:
        jscc = replace(jscc, device=str(args.device))
        motion = replace(motion, device=str(args.device))
    if getattr(args, "reconstruction_batch_size", None) is not None:
        motion = replace(
            motion,
            reconstruction_batch_size=args.reconstruction_batch_size,
        )
    if getattr(args, "metric_workers", None) is not None:
        reconstruction = replace(
            reconstruction,
            metric_workers=args.metric_workers,
        )
    return config, data, predictor, motion, jscc, reconstruction


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed
