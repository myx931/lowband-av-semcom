"""Small MLP JSCC model using complex Sionna PHY channel semantics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from av_semcom.channel.awgn import (
    ComplexAWGNChannel,
    masked_average_power,
    noise_power_from_snr_db,
    normalize_average_power,
    pack_real_symbols,
    unpack_complex_symbols,
)


@dataclass(frozen=True)
class JSCCForwardResult:
    """Decoded residual and diagnostics from one complex-channel forward pass."""

    decoded_residual: torch.Tensor
    transmitted_symbols: torch.Tensor
    received_symbols: torch.Tensor
    average_power: torch.Tensor
    noise_power: torch.Tensor


class ResidualJSCC(nn.Module):
    """Map one 18-D residual frame to a fixed number of complex channel uses."""

    def __init__(
        self,
        *,
        channel: ComplexAWGNChannel,
        input_dim: int = 18,
        hidden_dim: int = 64,
        channel_uses: int = 2,
        target_power: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim != 18:
            raise ValueError("input_dim must be 18")
        if hidden_dim < 1 or channel_uses < 1:
            raise ValueError("hidden_dim and channel_uses must be positive")
        if target_power <= 0:
            raise ValueError("target_power must be positive")
        self.channel = channel
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.channel_uses = channel_uses
        self.target_power = float(target_power)
        real_channel_dim = 2 * channel_uses
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, real_channel_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(real_channel_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(
        self,
        normalized_residual: torch.Tensor,
        valid_mask: torch.Tensor,
        snr_db: float | torch.Tensor,
        *,
        add_noise: bool = True,
        noise_seed: int | None = None,
    ) -> JSCCForwardResult:
        """Transmit ``[B,T,18]`` residuals through a complex AWGN channel."""

        if normalized_residual.ndim != 3 or normalized_residual.shape[-1] != self.input_dim:
            raise ValueError("normalized_residual must have shape [B,T,18]")
        if valid_mask.shape != normalized_residual.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean with shape [B,T]")
        if not torch.isfinite(normalized_residual).all():
            raise ValueError("normalized_residual contains non-finite values")
        encoded = pack_real_symbols(self.encoder(normalized_residual))
        transmitted, _ = normalize_average_power(
            encoded,
            valid_mask,
            target_power=self.target_power,
        )
        noise_power = noise_power_from_snr_db(
            snr_db,
            signal_power=self.target_power,
            like=transmitted,
        )
        received = (
            self.channel.transmit(
                transmitted,
                noise_power,
                noise_seed=noise_seed,
            )
            if add_noise
            else transmitted
        )
        decoded = self.decoder(unpack_complex_symbols(received))
        decoded = decoded * valid_mask.to(decoded.dtype).unsqueeze(-1)
        return JSCCForwardResult(
            decoded_residual=decoded,
            transmitted_symbols=transmitted,
            received_symbols=received,
            average_power=masked_average_power(transmitted, valid_mask),
            noise_power=noise_power,
        )


def masked_residual_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return MSE over valid residual coordinates only."""

    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 18:
        raise ValueError("prediction and target must share shape [B,T,18]")
    if valid_mask.shape != prediction.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean with shape [B,T]")
    weights = valid_mask.to(prediction.dtype).unsqueeze(-1).expand_as(prediction)
    denominator = weights.sum()
    if denominator.item() <= 0:
        raise ValueError("valid_mask contains no valid values")
    return (prediction.sub(target).square() * weights).sum() / denominator
