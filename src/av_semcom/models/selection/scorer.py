"""Small channel-aware hard Top-K residual scorer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ScorerForwardResult:
    """Scores, exact forward mask, and retained residual."""

    scores: torch.Tensor
    hard_mask: torch.Tensor
    selected_residual: torch.Tensor


@dataclass(frozen=True)
class PositionVelocityLoss:
    """Raw-motion position and temporal losses."""

    total: torch.Tensor
    position_l1: torch.Tensor
    velocity_l1: torch.Tensor


class ChannelAwareResidualScorer(nn.Module):
    """Score 18 residual dimensions from source residual, dynamics, SNR, and budget."""

    def __init__(
        self,
        *,
        motion_std: torch.Tensor,
        hidden_dim: int = 64,
        temperature: float = 1.0,
        max_channel_uses: int = 4,
    ) -> None:
        super().__init__()
        if motion_std.shape != (18,) or not torch.isfinite(motion_std).all():
            raise ValueError("motion_std must be finite with shape [18]")
        if torch.any(motion_std <= 0):
            raise ValueError("motion_std must be positive")
        if hidden_dim < 1 or temperature <= 0 or max_channel_uses < 1:
            raise ValueError("scorer dimensions and temperature must be positive")
        self.register_buffer("motion_std", motion_std.detach().float().clone())
        self.temperature = float(temperature)
        self.max_channel_uses = int(max_channel_uses)
        self.network = nn.Sequential(
            nn.Linear(18 * 3 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 18),
        )

    def forward(
        self,
        normalized_residual: torch.Tensor,
        transmission_mask: torch.Tensor,
        snr_db: float | torch.Tensor,
        *,
        k: int,
        channel_uses: int,
    ) -> ScorerForwardResult:
        """Select exactly K coordinates per eligible frame with straight-through Top-K."""

        _validate_residual_and_mask(normalized_residual, transmission_mask)
        if not 0 < k < 18:
            raise ValueError("k must be in [1,17]")
        if not 0 < channel_uses <= self.max_channel_uses:
            raise ValueError("channel_uses is outside the scorer range")
        raw = normalized_residual * self.motion_std
        delta = torch.zeros_like(raw)
        delta[:, 1:] = torch.abs(raw[:, 1:] - raw[:, :-1])
        snr_feature = (
            _broadcast_scalar_feature(
                snr_db,
                normalized_residual,
            )
            / 10.0
        )
        budget_feature = torch.full_like(snr_feature, k / 18.0)
        channel_feature = torch.full_like(
            snr_feature,
            channel_uses / self.max_channel_uses,
        )
        features = torch.cat(
            (
                normalized_residual,
                torch.abs(raw),
                delta,
                snr_feature,
                budget_feature,
                channel_feature,
            ),
            dim=-1,
        )
        scores = self.network(features)
        hard = hard_top_k_mask(scores, transmission_mask, k)
        soft = torch.softmax(scores / self.temperature, dim=-1) * k
        straight_through = hard + soft - soft.detach()
        straight_through = straight_through * transmission_mask.unsqueeze(-1)
        selected = normalized_residual * straight_through
        return ScorerForwardResult(
            scores=scores,
            hard_mask=hard.bool(),
            selected_residual=selected,
        )


def hard_top_k_mask(
    scores: torch.Tensor,
    transmission_mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Return a deterministic exact-K float mask with low-index tie preference."""

    if scores.ndim != 3 or scores.shape[-1] != 18:
        raise ValueError("scores must have shape [B,T,18]")
    if transmission_mask.shape != scores.shape[:2] or transmission_mask.dtype != torch.bool:
        raise ValueError("transmission_mask must be boolean with shape [B,T]")
    if not 0 <= k <= 18:
        raise ValueError("k must be in [0,18]")
    mask = torch.zeros_like(scores)
    if k:
        ranking = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :k]
        mask.scatter_(-1, ranking, 1.0)
    return mask * transmission_mask.to(scores.dtype).unsqueeze(-1)


def rule_selection_mask(
    normalized_residual: torch.Tensor,
    transmission_mask: torch.Tensor,
    *,
    k: int,
    method: str,
    motion_std: torch.Tensor,
    fixed_indices: torch.Tensor | None = None,
    random_seed: int | None = None,
) -> torch.Tensor:
    """Build exact-K masks for matched rule baselines."""

    _validate_residual_and_mask(normalized_residual, transmission_mask)
    if motion_std.shape != (18,):
        raise ValueError("motion_std must have shape [18]")
    if method == "normalized_magnitude":
        scores = torch.abs(normalized_residual)
    elif method == "raw_magnitude":
        scores = torch.abs(normalized_residual * motion_std)
    elif method == "fixed_train_magnitude":
        if fixed_indices is None or fixed_indices.shape != (k,):
            raise ValueError("fixed_train_magnitude requires fixed_indices with shape [k]")
        scores = torch.zeros_like(normalized_residual)
        weights = torch.arange(
            k,
            0,
            -1,
            dtype=scores.dtype,
            device=scores.device,
        )
        scores[..., fixed_indices.to(scores.device)] = weights
    elif method == "random":
        if random_seed is None:
            raise ValueError("random selection requires an explicit seed")
        generator = torch.Generator(device=normalized_residual.device).manual_seed(random_seed)
        scores = torch.rand(
            normalized_residual.shape,
            dtype=normalized_residual.dtype,
            device=normalized_residual.device,
            generator=generator,
        )
    else:
        raise ValueError(f"unsupported residual selection rule: {method}")
    return hard_top_k_mask(scores, transmission_mask, k).bool()


def raw_position_velocity_loss(
    decoded_normalized: torch.Tensor,
    target_normalized: torch.Tensor,
    valid_mask: torch.Tensor,
    motion_std: torch.Tensor,
    *,
    velocity_weight: float,
) -> PositionVelocityLoss:
    """Optimize raw-motion position and adjacent-valid-frame velocity L1."""

    if decoded_normalized.shape != target_normalized.shape:
        raise ValueError("decoded and target residuals must share shape")
    if valid_mask.shape != decoded_normalized.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean with shape [B,T]")
    if motion_std.shape != (18,) or velocity_weight < 0:
        raise ValueError("invalid motion_std or velocity_weight")
    raw_error = (decoded_normalized - target_normalized) * motion_std
    weights = valid_mask.to(raw_error.dtype).unsqueeze(-1).expand_as(raw_error)
    position = (torch.abs(raw_error) * weights).sum() / weights.sum()
    decoded_raw = decoded_normalized * motion_std
    target_raw = target_normalized * motion_std
    velocity_error = torch.diff(decoded_raw, dim=1) - torch.diff(target_raw, dim=1)
    pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
    pair_weights = pair_mask.to(raw_error.dtype).unsqueeze(-1).expand_as(velocity_error)
    if pair_weights.sum().item() <= 0:
        velocity = torch.zeros((), dtype=raw_error.dtype, device=raw_error.device)
    else:
        velocity = (torch.abs(velocity_error) * pair_weights).sum() / pair_weights.sum()
    return PositionVelocityLoss(
        total=position + velocity_weight * velocity,
        position_l1=position,
        velocity_l1=velocity,
    )


def _validate_residual_and_mask(
    residual: torch.Tensor,
    transmission_mask: torch.Tensor,
) -> None:
    if residual.ndim != 3 or residual.shape[-1] != 18:
        raise ValueError("normalized_residual must have shape [B,T,18]")
    if transmission_mask.shape != residual.shape[:2] or transmission_mask.dtype != torch.bool:
        raise ValueError("transmission_mask must be boolean with shape [B,T]")
    if not torch.isfinite(residual).all():
        raise ValueError("normalized_residual contains non-finite values")


def _broadcast_scalar_feature(
    value: float | torch.Tensor,
    like: torch.Tensor,
) -> torch.Tensor:
    feature = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    while feature.ndim < 3:
        feature = feature.unsqueeze(-1)
    try:
        return torch.broadcast_to(feature, (*like.shape[:2], 1))
    except RuntimeError as exc:
        raise ValueError("scalar feature is not broadcastable to [B,T,1]") from exc
