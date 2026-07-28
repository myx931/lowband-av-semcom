"""CLI configuration for E7 full-motion JSCC commands."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from av_semcom.data.grid import GridSettings
from av_semcom.models.full_motion.config import full_motion_jscc_settings
from av_semcom.models.jscc.config import JSCCSettings
from av_semcom.models.predictor.config import AudioMotionSettings
from av_semcom.utils.config import Config, load_yaml_config


def build_full_motion_parser(
    description: str,
    *,
    evaluation: bool = False,
) -> argparse.ArgumentParser:
    """Build the train/evaluate parser without manifest resampling."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/residual_jscc_ten_speaker.yaml"),
    )
    parser.add_argument("--source-e5-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=evaluation)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device")
    return parser


def settings_from_args(
    args: argparse.Namespace,
) -> tuple[Config, AudioMotionSettings, JSCCSettings]:
    """Resolve the frozen motion statistics and E7 JSCC settings."""

    config = load_yaml_config(args.config)
    data = GridSettings.from_config(config)
    predictor = AudioMotionSettings.from_config(config, data)
    jscc = full_motion_jscc_settings(config)
    if args.device:
        jscc = replace(jscc, device=str(args.device))
    return config, predictor, jscc
