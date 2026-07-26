"""Tests for typed motion artifacts and pilot normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from av_semcom.models.motion.perturbations import (
    fit_motion_normalizer,
    load_motion_normalizer,
    save_motion_normalizer,
)
from av_semcom.models.motion.sequence import (
    load_motion_sequence,
    save_motion_sequence,
)
from av_semcom.models.reconstruction.backend import FakeReconstructionBackend


def _sequence(frame_count: int = 5):
    crops = np.zeros((frame_count, 256, 256, 3), dtype=np.uint8)
    for index in range(frame_count):
        crops[index].fill(index * 20)
    return FakeReconstructionBackend().extract_motion(
        crops,
        np.ones(frame_count, dtype=np.bool_),
        sample_id="s1_example",
        fps=25,
        config_fingerprint="abc",
    )


def test_motion_sequence_round_trip_and_shapes(tmp_path: Path) -> None:
    sequence = _sequence()
    path = tmp_path / "motion.npz"

    save_motion_sequence(path, sequence)
    loaded = load_motion_sequence(path)

    assert loaded.sample_id == sequence.sample_id
    assert loaded.expression.shape == (5, 21, 3)
    assert loaded.lip_delta.shape == (5, 6, 3)
    assert loaded.lip_vector.shape == (5, 18)
    assert loaded.rotation.shape == (5, 3, 3)
    assert loaded.translation.shape == (5, 3)
    assert loaded.scale.shape == (5, 1)
    assert loaded.canonical_keypoints.shape == (5, 21, 3)
    assert np.allclose(loaded.lip_delta[loaded.source_frame_index], 0)


def test_motion_normalizer_round_trip_and_json(tmp_path: Path) -> None:
    sequence = _sequence()
    normalizer = fit_motion_normalizer([sequence])
    normalized = normalizer.normalize(sequence.lip_vector)

    assert np.allclose(normalizer.denormalize(normalized), sequence.lip_vector)

    path = tmp_path / "stats.json"
    save_motion_normalizer(path, normalizer)
    loaded = load_motion_normalizer(path)
    assert loaded.scope == "pilot_stats"
    assert np.allclose(loaded.mean, normalizer.mean)
    assert np.allclose(loaded.std, normalizer.std)


def test_motion_normalizer_records_train_only_scope() -> None:
    normalizer = fit_motion_normalizer([_sequence()], scope="train_stats")

    assert normalizer.scope == "train_stats"
