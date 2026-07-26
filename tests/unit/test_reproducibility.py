"""Tests for experiment seed initialization."""

import os

import numpy as np
import pytest
import torch

from av_semcom.utils.reproducibility import seed_everything


def test_seed_everything_repeats_random_streams() -> None:
    seed_everything(7)
    numpy_first = np.random.rand(3)
    torch_first = torch.rand(3)

    seed_everything(7)

    assert np.array_equal(numpy_first, np.random.rand(3))
    assert torch.equal(torch_first, torch.rand(3))


def test_seed_everything_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        seed_everything(-1)


def test_deterministic_seed_sets_cublas_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    seed_everything(42, deterministic=True)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
