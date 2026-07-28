"""Shape, power, noise, and loss tests for the lightweight residual JSCC."""

from __future__ import annotations

import pytest
import torch

from av_semcom.channel.awgn import NativeComplexAWGN
from av_semcom.models.jscc.model import ResidualJSCC, masked_residual_mse


def test_jscc_forward_shapes_power_mask_and_determinism() -> None:
    torch.manual_seed(3)
    model = ResidualJSCC(
        channel=NativeComplexAWGN(seed=3),
        hidden_dim=16,
        channel_uses=2,
    )
    residual = torch.randn((2, 5, 18))
    mask = torch.tensor([[False, True, True, True, False], [False, True, True, True, True]])

    first = model(
        residual,
        mask,
        torch.tensor([0.0, 10.0]),
        noise_seed=42,
    )
    second = model(
        residual,
        mask,
        torch.tensor([0.0, 10.0]),
        noise_seed=42,
    )

    assert first.decoded_residual.shape == (2, 5, 18)
    assert first.transmitted_symbols.shape == (2, 5, 2)
    assert first.transmitted_symbols.is_complex()
    assert torch.allclose(first.average_power, torch.ones(2), atol=1e-6)
    assert torch.equal(first.received_symbols, second.received_symbols)
    assert torch.equal(first.decoded_residual[:, 0], torch.zeros((2, 18)))
    assert torch.equal(first.decoded_residual[0, 4], torch.zeros(18))


def test_jscc_noiseless_path_and_masked_mse() -> None:
    model = ResidualJSCC(
        channel=NativeComplexAWGN(seed=5),
        hidden_dim=8,
        channel_uses=1,
    )
    residual = torch.zeros((1, 3, 18))
    residual[0, 1] = 1
    mask = torch.tensor([[False, True, False]])

    result = model(residual, mask, 5.0, add_noise=False)
    loss = masked_residual_mse(result.decoded_residual, residual, mask)

    assert torch.equal(result.received_symbols, result.transmitted_symbols)
    assert loss.ndim == 0
    with pytest.raises(ValueError, match="no valid"):
        masked_residual_mse(residual, residual, torch.zeros_like(mask))
