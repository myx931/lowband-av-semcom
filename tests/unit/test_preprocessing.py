"""Tests for resumable and interpolation primitives."""

from pathlib import Path

import numpy as np
import pytest

from av_semcom.data.preprocessing import (
    StaleArtifactError,
    atomic_save_npz,
    config_fingerprint,
    interpolate_missing,
    should_process,
    write_artifact_metadata,
)


def test_matching_artifact_is_skipped(tmp_path: Path) -> None:
    output = tmp_path / "features.npz"
    fingerprint = config_fingerprint({"stage": "audio", "n_mels": 80})
    atomic_save_npz(output, features=np.zeros((2, 3), dtype=np.float32))
    write_artifact_metadata(output, fingerprint)

    assert not should_process(output, fingerprint)


def test_stale_artifact_requires_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "features.npz"
    atomic_save_npz(output, features=np.zeros((1,), dtype=np.float32))
    write_artifact_metadata(output, config_fingerprint({"version": 1}))

    with pytest.raises(StaleArtifactError, match="different configuration"):
        should_process(output, config_fingerprint({"version": 2}))
    assert should_process(output, config_fingerprint({"version": 2}), overwrite=True)


def test_interpolate_missing_fills_middle_and_edges() -> None:
    values = np.asarray([[0.0], [np.nan], [2.0], [np.nan]], dtype=np.float32)
    valid = np.asarray([True, False, True, False])

    result = interpolate_missing(values, valid)

    assert np.array_equal(result[:, 0], np.asarray([0.0, 1.0, 2.0, 2.0]))
