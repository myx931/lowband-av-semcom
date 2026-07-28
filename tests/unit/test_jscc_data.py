from __future__ import annotations

import numpy as np
import pytest

from av_semcom.models.jscc.data import (
    ResidualDataset,
    ResidualExample,
    load_residual_example,
    save_residual_example,
)


def _example() -> ResidualExample:
    target = np.zeros((5, 18), dtype=np.float32)
    target[1:] = 2
    prediction = np.zeros_like(target)
    prediction[1:] = 1
    valid = np.asarray([True, True, True, False, True], dtype=np.bool_)
    transmission = valid.copy()
    transmission[0] = False
    raw = target - prediction
    raw[~transmission] = 0
    normalized = raw / np.float32(2)
    return ResidualExample(
        sample_id="s1_demo",
        speaker_id="s1",
        split="train",
        prediction=prediction,
        target=target,
        raw_residual=raw,
        normalized_residual=normalized,
        valid_mask=valid,
        transmission_mask=transmission,
    )


def test_residual_example_excludes_reference_and_invalid_frames() -> None:
    example = _example()
    item = ResidualDataset([example])[0]

    assert example.transmission_mask.tolist() == [False, True, True, False, True]
    assert item["residual"].shape == (5, 18)  # type: ignore[union-attr]
    assert not item["mask"][0]  # type: ignore[index]


def test_residual_cache_round_trip_and_fingerprint(tmp_path) -> None:
    path = tmp_path / "example.npz"
    save_residual_example(path, _example(), experiment_fingerprint="correct")

    restored = load_residual_example(path, expected_fingerprint="correct")

    assert restored.sample_id == "s1_demo"
    assert np.array_equal(restored.normalized_residual, _example().normalized_residual)
    with pytest.raises(ValueError, match="fingerprint"):
        load_residual_example(path, expected_fingerprint="stale")


def test_residual_example_rejects_reference_transmission() -> None:
    example = _example()
    invalid_mask = example.transmission_mask.copy()
    invalid_mask[0] = True

    with pytest.raises(ValueError, match="transmission_mask"):
        ResidualExample(
            **{
                **example.__dict__,
                "transmission_mask": invalid_mask,
            }
        )
