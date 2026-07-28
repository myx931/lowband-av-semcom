from __future__ import annotations

import numpy as np
import pytest
import torch

from av_semcom.models.jscc.config import JSCCSettings
from av_semcom.models.jscc.data import ResidualExample
from av_semcom.models.jscc.experiment import _evaluate_one_model, _train_one_model

pytestmark = pytest.mark.smoke


def _settings(tmp_path) -> JSCCSettings:
    return JSCCSettings.from_config(
        {
            "channel": {
                "backend": "native_reference",
                "complex_channel_uses": [1],
                "target_power": 1.0,
            },
            "jscc_model": {"input_dim": 18, "hidden_dim": 8},
            "jscc_training": {
                "seeds": [42],
                "device": "cpu",
                "batch_size": 2,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "max_epochs": 2,
                "early_stopping_patience": 2,
                "early_stopping_min_delta": 0.0,
                "gradient_clip_norm": 1.0,
                "num_workers": 0,
                "deterministic": True,
                "snr_min_db": 0.0,
                "snr_max_db": 5.0,
            },
            "jscc_evaluation": {
                "output_dir": str(tmp_path / "output"),
                "validation_snr_db": [2.5],
                "test_snr_db": [0.0],
                "noise_seeds": [42],
            },
        }
    )


def _examples(split: str, speaker: str, count: int) -> list[ResidualExample]:
    examples = []
    for index in range(count):
        generator = np.random.default_rng(index)
        normalized = generator.normal(size=(6, 18)).astype(np.float32)
        mask = np.ones(6, dtype=np.bool_)
        transmission = mask.copy()
        transmission[0] = False
        normalized[0] = 0
        prediction = np.zeros_like(normalized)
        target = normalized * np.float32(0.1)
        examples.append(
            ResidualExample(
                sample_id=f"{speaker}_{index}",
                speaker_id=speaker,
                split=split,
                prediction=prediction,
                target=target,
                raw_residual=target.copy(),
                normalized_residual=normalized,
                valid_mask=mask,
                transmission_mask=transmission,
            )
        )
    return examples


def test_synthetic_residual_jscc_train_and_evaluate(tmp_path) -> None:
    settings = _settings(tmp_path)
    train = _examples("train", "s1", 4)
    validation = _examples("validation", "s2", 2)
    model_dir = tmp_path / "model"

    result = _train_one_model(
        settings,
        train,
        validation,
        channel_uses=1,
        seed=42,
        model_dir=model_dir,
        fingerprint="smoke",
    )

    assert result["best_epoch"] in {1, 2}
    checkpoint = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=False)
    from av_semcom.channel.awgn import NativeComplexAWGN
    from av_semcom.models.jscc.model import ResidualJSCC

    model = ResidualJSCC(
        channel=NativeComplexAWGN(seed=42),
        input_dim=18,
        hidden_dim=8,
        channel_uses=1,
    )
    model.load_state_dict(checkpoint["model_state"])
    rows = _evaluate_one_model(
        model,
        validation,
        settings,
        np.full(18, 0.1, dtype=np.float32),
        channel_uses=1,
        model_seed=42,
        device=torch.device("cpu"),
    )

    assert len(rows) == 4
    assert {row["condition"] for row in rows} == {
        "noiseless_autoencoder",
        "jscc_awgn",
    }
    assert all(np.isfinite(row["l1"]) for row in rows)
