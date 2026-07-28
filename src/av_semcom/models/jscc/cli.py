"""Command-line configuration shared by E5 residual JSCC commands."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from av_semcom.data.grid import GridSettings
from av_semcom.models.jscc.config import JSCCSettings
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
