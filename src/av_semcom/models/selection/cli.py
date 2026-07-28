"""Command-line configuration for E6 channel-aware selection baselines."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from av_semcom.data.grid import GridSettings
from av_semcom.models.jscc.config import JSCCSettings
from av_semcom.models.predictor.config import AudioMotionSettings
from av_semcom.models.selection.config import (
    ChannelGateSettings,
    ResidualScorerAblationSettings,
    ResidualScorerSettings,
)
from av_semcom.utils.config import Config, load_yaml_config


def build_channel_gate_parser(description: str) -> argparse.ArgumentParser:
    """Build the validation-calibration and frozen-test CLI."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/residual_jscc_ten_speaker.yaml"),
    )
    parser.add_argument("--e5-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", help="Override jscc_training.device.")
    return parser


def channel_gate_settings_from_args(
    args: argparse.Namespace,
) -> tuple[
    Config,
    GridSettings,
    AudioMotionSettings,
    JSCCSettings,
    ChannelGateSettings,
]:
    """Resolve the frozen E5 source and the independent E6 gate settings."""

    config = load_yaml_config(args.config)
    data = GridSettings.from_config(config)
    predictor = AudioMotionSettings.from_config(config, data)
    jscc = JSCCSettings.from_config(config)
    if args.device:
        jscc = replace(jscc, device=str(args.device))
    gate = ChannelGateSettings.from_config(config, jscc)
    if args.resume and args.run_dir is None:
        raise ValueError("--resume requires --run-dir")
    return config, data, predictor, jscc, gate


def build_residual_scorer_parser(
    description: str,
    *,
    evaluation: bool = False,
) -> argparse.ArgumentParser:
    """Build the scorer-only training or frozen test parser."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/residual_jscc_ten_speaker.yaml"),
    )
    parser.add_argument("--e5-run-dir", type=Path, required=True)
    parser.add_argument("--gate-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=evaluation)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device")
    return parser


def residual_scorer_settings_from_args(
    args: argparse.Namespace,
) -> tuple[
    Config,
    GridSettings,
    AudioMotionSettings,
    JSCCSettings,
    ResidualScorerSettings,
]:
    """Resolve the shared E5 source and E6 scorer settings."""

    config = load_yaml_config(args.config)
    data = GridSettings.from_config(config)
    predictor = AudioMotionSettings.from_config(config, data)
    jscc = JSCCSettings.from_config(config)
    scorer = ResidualScorerSettings.from_config(config, jscc)
    if args.device:
        jscc = replace(jscc, device=str(args.device))
        scorer = replace(scorer, device=str(args.device))
    if args.resume and args.run_dir is None:
        raise ValueError("--resume requires --run-dir")
    return config, data, predictor, jscc, scorer


def build_residual_scorer_ablation_parser(
    description: str,
    *,
    evaluation: bool = False,
) -> argparse.ArgumentParser:
    """Build the validation-only scorer ablation CLI."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/residual_jscc_ten_speaker.yaml"),
    )
    parser.add_argument("--e5-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=evaluation)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device")
    return parser


def residual_scorer_ablation_settings_from_args(
    args: argparse.Namespace,
) -> tuple[
    Config,
    GridSettings,
    AudioMotionSettings,
    JSCCSettings,
    ResidualScorerSettings,
    ResidualScorerAblationSettings,
]:
    """Resolve E5, scorer, and validation-only ablation settings."""

    config = load_yaml_config(args.config)
    data = GridSettings.from_config(config)
    predictor = AudioMotionSettings.from_config(config, data)
    jscc = JSCCSettings.from_config(config)
    scorer = ResidualScorerSettings.from_config(config, jscc)
    ablation = ResidualScorerAblationSettings.from_config(config, jscc, scorer)
    if args.device:
        jscc = replace(jscc, device=str(args.device))
        scorer = replace(scorer, device=str(args.device))
    if args.resume and args.run_dir is None:
        raise ValueError("--resume requires --run-dir")
    return config, data, predictor, jscc, scorer, ablation
