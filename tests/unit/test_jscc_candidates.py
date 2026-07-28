from __future__ import annotations

import numpy as np
import pytest

from av_semcom.models.jscc.candidates import (
    JSCCCandidateBundle,
    JSCCCondition,
    condition_id,
    load_candidate_bundle,
    save_candidate_bundle,
)
from av_semcom.models.jscc.export import select_validation_model_seeds


def _conditions() -> tuple[JSCCCondition, ...]:
    return (
        JSCCCondition(
            condition_id="prediction_only",
            family="prediction_only",
            channel_uses=None,
            model_seed=None,
            snr_db=None,
            noise_seed=None,
        ),
        JSCCCondition(
            condition_id=condition_id(
                "jscc_awgn",
                channel_uses=2,
                model_seed=43,
                snr_db=5.0,
                noise_seed=42,
            ),
            family="jscc_awgn",
            channel_uses=2,
            model_seed=43,
            snr_db=5.0,
            noise_seed=42,
        ),
    )


def test_candidate_bundle_round_trip(tmp_path) -> None:
    vectors = np.zeros((2, 5, 18), dtype=np.float32)
    vectors[:, 1:] = 0.5
    bundle = JSCCCandidateBundle(
        sample_id="s7_demo",
        split="test",
        speaker_id="s7",
        experiment_fingerprint="experiment",
        candidate_fingerprint="candidate",
        conditions=_conditions(),
        vectors=vectors,
        valid_mask=np.ones(5, dtype=np.bool_),
    )
    path = tmp_path / "bundle.npz"

    save_candidate_bundle(path, bundle)
    restored = load_candidate_bundle(path, expected_fingerprint="candidate")

    assert restored.conditions == bundle.conditions
    assert np.array_equal(restored.vector("prediction_only"), vectors[0])
    with pytest.raises(ValueError, match="fingerprint"):
        load_candidate_bundle(path, expected_fingerprint="stale")


def test_validation_model_selection_is_per_channel_use() -> None:
    summary = {
        "models": [
            {
                "channel_uses": 1,
                "seed": 42,
                "best_validation_normalized_mse": 0.5,
            },
            {
                "channel_uses": 1,
                "seed": 43,
                "best_validation_normalized_mse": 0.4,
            },
            {
                "channel_uses": 2,
                "seed": 42,
                "best_validation_normalized_mse": 0.3,
            },
            {
                "channel_uses": 2,
                "seed": 43,
                "best_validation_normalized_mse": 0.35,
            },
        ]
    }

    assert select_validation_model_seeds(summary, (1, 2)) == {1: 43, 2: 42}
