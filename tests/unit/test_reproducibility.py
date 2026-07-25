"""Tests for experiment seed initialization."""

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
