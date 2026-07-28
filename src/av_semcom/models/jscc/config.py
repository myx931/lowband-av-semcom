"""Configuration for the E5 residual JSCC baseline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from av_semcom.utils.config import ConfigError


@dataclass(frozen=True)
class JSCCSettings:
    """Resolved channel, model, training, and evaluation settings."""

    output_root: Path
    channel_backend: str
    channel_uses: tuple[int, ...]
    target_power: float
    input_dim: int
    hidden_dim: int
    seeds: tuple[int, ...]
    device: str
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    gradient_clip_norm: float
    num_workers: int
    deterministic: bool
    train_snr_min_db: float
    train_snr_max_db: float
    validation_snr_db: tuple[float, ...]
    test_snr_db: tuple[float, ...]
    noise_seeds: tuple[int, ...]
    config: Mapping[str, Any]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> JSCCSettings:
        """Validate the deliberately small E5 experiment surface."""

        channel = _mapping(config, "channel")
        model = _mapping(config, "jscc_model")
        training = _mapping(config, "jscc_training")
        evaluation = _mapping(config, "jscc_evaluation")

        backend = str(channel.get("backend", "sionna"))
        if backend not in {"sionna", "native_reference"}:
            raise ConfigError("channel.backend must be sionna or native_reference")
        uses = _unique_positive_ints(channel, "complex_channel_uses")
        target_power = _positive_float(channel, "target_power", 1.0)
        input_dim = _positive_int(model, "input_dim", 18)
        if input_dim != 18:
            raise ConfigError("jscc_model.input_dim must be 18")
        train_min = float(training.get("snr_min_db", 0.0))
        train_max = float(training.get("snr_max_db", 10.0))
        if not train_min < train_max:
            raise ConfigError("jscc_training SNR range must have min < max")
        validation_snr = _unique_floats(evaluation, "validation_snr_db")
        test_snr = _unique_floats(evaluation, "test_snr_db")
        if set(validation_snr) & set(test_snr):
            raise ConfigError("validation and test SNR grids must be disjoint")
        seeds = _unique_nonnegative_ints(training, "seeds")
        noise_seeds = _unique_nonnegative_ints(evaluation, "noise_seeds")
        output_raw = evaluation.get("output_dir", "outputs/residual_jscc")
        if not isinstance(output_raw, str) or not output_raw:
            raise ConfigError("jscc_evaluation.output_dir must be a non-empty path")
        output_root = Path(output_raw)
        if not output_root.is_absolute():
            output_root = Path(__file__).resolve().parents[4] / output_root
        return cls(
            output_root=output_root.resolve(),
            channel_backend=backend,
            channel_uses=uses,
            target_power=target_power,
            input_dim=input_dim,
            hidden_dim=_positive_int(model, "hidden_dim", 64),
            seeds=seeds,
            device=str(training.get("device", "cuda:0")),
            batch_size=_positive_int(training, "batch_size", 16),
            learning_rate=_positive_float(training, "learning_rate", 1e-3),
            weight_decay=_nonnegative_float(training, "weight_decay", 1e-4),
            max_epochs=_positive_int(training, "max_epochs", 100),
            early_stopping_patience=_positive_int(
                training,
                "early_stopping_patience",
                15,
            ),
            early_stopping_min_delta=_nonnegative_float(
                training,
                "early_stopping_min_delta",
                1e-4,
            ),
            gradient_clip_norm=_positive_float(training, "gradient_clip_norm", 1.0),
            num_workers=int(training.get("num_workers", 0)),
            deterministic=bool(training.get("deterministic", True)),
            train_snr_min_db=train_min,
            train_snr_max_db=train_max,
            validation_snr_db=validation_snr,
            test_snr_db=test_snr,
            noise_seeds=noise_seeds,
            config={key: value for key, value in config.items() if key != "jscc_reconstruction"},
        )

    def require_formal_backend(self) -> None:
        """Reject the reference implementation for a reported experiment."""

        if self.channel_backend != "sionna":
            raise ConfigError("formal E5 runs require channel.backend=sionna")


@dataclass(frozen=True)
class JSCCReconstructionSettings:
    """Frozen subset of E5 conditions used for video reconstruction."""

    split: str
    noise_seed: int
    metric_workers: int
    media_channel_uses: int
    save_representative_media: bool

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        jscc: JSCCSettings,
    ) -> JSCCReconstructionSettings:
        """Validate that video evaluation cannot silently tune on test."""

        reconstruction = _mapping(config, "jscc_reconstruction")
        split = str(reconstruction.get("split", "test"))
        if split != "test":
            raise ConfigError("jscc_reconstruction.split must be test")
        noise_seed = int(reconstruction.get("noise_seed", 42))
        if noise_seed not in jscc.noise_seeds:
            raise ConfigError("jscc_reconstruction.noise_seed must be an evaluation noise seed")
        media_channel_uses = int(reconstruction.get("media_channel_uses", max(jscc.channel_uses)))
        if media_channel_uses not in jscc.channel_uses:
            raise ConfigError(
                "jscc_reconstruction.media_channel_uses must be a configured channel use"
            )
        metric_workers = int(reconstruction.get("metric_workers", 4))
        if metric_workers < 1:
            raise ConfigError("jscc_reconstruction.metric_workers must be positive")
        return cls(
            split=split,
            noise_seed=noise_seed,
            metric_workers=metric_workers,
            media_channel_uses=media_channel_uses,
            save_representative_media=bool(reconstruction.get("save_representative_media", True)),
        )


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} configuration must be a mapping")
    return value


def _positive_int(config: Mapping[str, Any], key: str, default: int) -> int:
    value = int(config.get(key, default))
    if value <= 0:
        raise ConfigError(f"{key} must be positive")
    return value


def _positive_float(config: Mapping[str, Any], key: str, default: float) -> float:
    value = float(config.get(key, default))
    if value <= 0:
        raise ConfigError(f"{key} must be positive")
    return value


def _nonnegative_float(config: Mapping[str, Any], key: str, default: float) -> float:
    value = float(config.get(key, default))
    if value < 0:
        raise ConfigError(f"{key} must be non-negative")
    return value


def _unique_positive_ints(config: Mapping[str, Any], key: str) -> tuple[int, ...]:
    raw = config.get(key)
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{key} must be a non-empty list")
    values = tuple(int(value) for value in raw)
    if values != tuple(sorted(set(values))) or any(value <= 0 for value in values):
        raise ConfigError(f"{key} must be sorted, unique, and positive")
    return values


def _unique_nonnegative_ints(config: Mapping[str, Any], key: str) -> tuple[int, ...]:
    raw = config.get(key)
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{key} must be a non-empty list")
    values = tuple(int(value) for value in raw)
    if len(values) != len(set(values)) or any(value < 0 for value in values):
        raise ConfigError(f"{key} must contain unique non-negative integers")
    return values


def _unique_floats(config: Mapping[str, Any], key: str) -> tuple[float, ...]:
    raw = config.get(key)
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{key} must be a non-empty list")
    values = tuple(float(value) for value in raw)
    if values != tuple(sorted(set(values))):
        raise ConfigError(f"{key} must be sorted and unique")
    return values
