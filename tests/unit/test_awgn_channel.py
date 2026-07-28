"""Unit tests for complex AWGN semantics shared with Sionna PHY."""

from __future__ import annotations

import math

import pytest
import torch

from av_semcom.channel.awgn import (
    NativeComplexAWGN,
    masked_average_power,
    noise_power_from_snr_db,
    normalize_average_power,
    pack_real_symbols,
    unpack_complex_symbols,
)


def test_real_complex_packing_round_trip() -> None:
    values = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8)

    symbols = pack_real_symbols(values)
    restored = unpack_complex_symbols(symbols)

    assert symbols.shape == (2, 3, 4)
    assert symbols.dtype == torch.complex64
    assert torch.equal(restored, values)
    with pytest.raises(ValueError, match="2C"):
        pack_real_symbols(torch.ones((1, 2, 3)))


def test_power_normalization_is_per_sample_and_masked() -> None:
    symbols = torch.tensor(
        [
            [[1 + 2j, 3 + 4j], [2 + 1j, 4 + 3j], [100 + 100j, 100 + 100j]],
            [[2 + 0j, 0 + 2j], [2 + 2j, 1 + 1j], [3 + 1j, 2 + 4j]],
        ],
        dtype=torch.complex64,
    )
    mask = torch.tensor([[True, True, False], [True, True, True]])

    normalized, scale = normalize_average_power(symbols, mask)

    assert scale.shape == (2,)
    assert torch.allclose(masked_average_power(normalized, mask), torch.ones(2))
    assert torch.equal(normalized[0, 2], torch.zeros(2))


def test_complex_awgn_uses_sionna_noise_power_convention_and_seed() -> None:
    symbols = torch.ones((2, 200_000, 1), dtype=torch.complex64)
    channel = NativeComplexAWGN(seed=7)
    noise_power = noise_power_from_snr_db(torch.tensor([0.0, 10.0]), like=symbols)

    first = channel.transmit(symbols, noise_power, noise_seed=42)
    repeated = channel.transmit(symbols, noise_power, noise_seed=42)
    noise = first - symbols

    assert torch.equal(first, repeated)
    assert noise_power.tolist() == pytest.approx([1.0, 0.1])
    assert noise[0].abs().square().mean().item() == pytest.approx(1.0, rel=0.02)
    assert noise[1].abs().square().mean().item() == pytest.approx(0.1, rel=0.02)
    assert noise[0].real.var().item() == pytest.approx(0.5, rel=0.02)
    assert noise[0].imag.var().item() == pytest.approx(0.5, rel=0.02)
    assert noise_power_from_snr_db(-5.0).item() == pytest.approx(math.pow(10, 0.5))


def test_power_normalization_rejects_invalid_inputs() -> None:
    symbols = torch.ones((1, 3, 2), dtype=torch.complex64)
    mask = torch.zeros((1, 3), dtype=torch.bool)

    with pytest.raises(ValueError, match="at least one valid"):
        normalize_average_power(symbols, mask)
    with pytest.raises(ValueError, match="complex"):
        normalize_average_power(symbols.real, ~mask)
