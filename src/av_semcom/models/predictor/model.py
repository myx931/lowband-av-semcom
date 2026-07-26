"""Causal GRU and masked training loss for audio-to-motion prediction."""

from __future__ import annotations

import torch
from torch import nn


class AudioToMotionGRU(nn.Module):
    """Map each 40 ms log-Mel group to an 18-D causal mouth-motion sequence."""

    def __init__(
        self,
        *,
        mel_bins: int = 80,
        mel_steps_per_frame: int = 4,
        audio_projection_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 18,
    ) -> None:
        super().__init__()
        if (
            min(
                mel_bins,
                mel_steps_per_frame,
                audio_projection_dim,
                hidden_dim,
                num_layers,
                output_dim,
            )
            <= 0
        ):
            raise ValueError("model dimensions must be positive")
        if output_dim != 18:
            raise ValueError("output_dim must be 18")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.mel_bins = mel_bins
        self.mel_steps_per_frame = mel_steps_per_frame
        self.output_dim = output_dim
        self.audio_encoder = nn.Sequential(
            nn.Linear(mel_bins * mel_steps_per_frame, audio_projection_dim),
            nn.LayerNorm(audio_projection_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=audio_projection_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False,
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Predict normalized motion from ``[B,T,4,80]`` audio."""

        if audio.ndim != 4 or audio.shape[2:] != (
            self.mel_steps_per_frame,
            self.mel_bins,
        ):
            raise ValueError(
                f"audio must have shape [B,T,{self.mel_steps_per_frame},{self.mel_bins}]"
            )
        batch_size, frame_count = audio.shape[:2]
        flattened = audio.reshape(batch_size, frame_count, -1)
        encoded = self.audio_encoder(flattened)
        hidden, _ = self.gru(encoded)
        prediction: torch.Tensor = self.output(hidden)
        return prediction


def masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return mean absolute error across valid frames and coordinates."""

    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 18:
        raise ValueError("prediction and target must share shape [B,T,18]")
    if mask.shape != prediction.shape[:2]:
        raise ValueError("mask must have shape [B,T]")
    weights = mask.to(dtype=prediction.dtype).unsqueeze(-1).expand_as(prediction)
    denominator = weights.sum()
    if denominator.item() <= 0:
        raise ValueError("mask contains no valid values")
    return (torch.abs(prediction - target) * weights).sum() / denominator
