"""Cross-check the formal Sionna PHY AWGN backend when its environment is active."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from av_semcom.channel.awgn import SionnaComplexAWGN

pytestmark = pytest.mark.integration


@pytest.mark.skipif(importlib.util.find_spec("sionna") is None, reason="Sionna is not installed")
def test_sionna_awgn_matches_documented_complex_noise_power() -> None:
    channel = SionnaComplexAWGN(device="cpu", seed=42)
    symbols = torch.ones((1, 200_000), dtype=torch.complex64)

    received = channel.transmit(symbols, torch.tensor(0.2), noise_seed=43)
    noise = received - symbols

    assert noise.abs().square().mean().item() == pytest.approx(0.2, rel=0.02)
    assert noise.real.var().item() == pytest.approx(0.1, rel=0.02)
    assert noise.imag.var().item() == pytest.approx(0.1, rel=0.02)
