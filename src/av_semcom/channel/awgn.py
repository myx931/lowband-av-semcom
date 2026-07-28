"""Complex AWGN adapters with Sionna as the formal experiment backend."""

from __future__ import annotations

from typing import Protocol

import torch


class ComplexAWGNChannel(Protocol):
    """Minimal interface shared by Sionna and the unit-test reference channel."""

    backend_name: str

    def transmit(
        self,
        symbols: torch.Tensor,
        noise_power: float | torch.Tensor,
        *,
        noise_seed: int | None = None,
    ) -> torch.Tensor:
        """Return noisy complex symbols with the same shape."""


def _validate_complex_symbols(symbols: torch.Tensor, valid_mask: torch.Tensor) -> None:
    if symbols.ndim != 3 or not symbols.is_complex():
        raise ValueError("symbols must be complex with shape [B,T,C]")
    if symbols.shape[-1] < 1:
        raise ValueError("symbols must contain at least one channel use")
    if valid_mask.shape != symbols.shape[:2]:
        raise ValueError("valid_mask must have shape [B,T]")
    if valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean")
    if not torch.isfinite(symbols).all():
        raise ValueError("symbols contain non-finite values")
    if torch.any(valid_mask.sum(dim=1) == 0):
        raise ValueError("every sample must contain at least one valid frame")


def pack_real_symbols(values: torch.Tensor) -> torch.Tensor:
    """Pack a final real dimension ``2C`` into ``C`` complex channel uses."""

    if values.ndim != 3 or values.shape[-1] < 2 or values.shape[-1] % 2:
        raise ValueError("values must have shape [B,T,2C]")
    if not torch.is_floating_point(values):
        raise ValueError("values must be floating point")
    pairs = values.reshape(*values.shape[:-1], values.shape[-1] // 2, 2).contiguous()
    return torch.view_as_complex(pairs)


def unpack_complex_symbols(symbols: torch.Tensor) -> torch.Tensor:
    """Unpack ``C`` complex channel uses into a final real dimension ``2C``."""

    if symbols.ndim != 3 or not symbols.is_complex():
        raise ValueError("symbols must be complex with shape [B,T,C]")
    pairs = torch.view_as_real(symbols)
    return pairs.reshape(*symbols.shape[:-1], symbols.shape[-1] * 2)


def masked_average_power(symbols: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Return per-sample average ``|x|²`` over valid complex channel uses."""

    _validate_complex_symbols(symbols, valid_mask)
    weights = valid_mask.to(symbols.real.dtype).unsqueeze(-1)
    scalar_count = valid_mask.sum(dim=1).to(symbols.real.dtype) * symbols.shape[-1]
    energy = (symbols.abs().square() * weights).sum(dim=(1, 2))
    return energy / scalar_count


def normalize_average_power(
    symbols: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    target_power: float = 1.0,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize each sequence codeword to target mean complex-symbol power."""

    _validate_complex_symbols(symbols, valid_mask)
    if target_power <= 0:
        raise ValueError("target_power must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    current_power = masked_average_power(symbols, valid_mask)
    target = torch.as_tensor(
        target_power,
        dtype=symbols.real.dtype,
        device=symbols.device,
    )
    scale = torch.sqrt(target / torch.clamp(current_power, min=epsilon))
    normalized = symbols * scale[:, None, None]
    normalized = normalized * valid_mask.to(symbols.real.dtype).unsqueeze(-1)
    return normalized, scale


def noise_power_from_snr_db(
    snr_db: float | torch.Tensor,
    *,
    signal_power: float = 1.0,
    like: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert SNR to complex noise power ``N₀ = P / 10^(SNR/10)``."""

    if signal_power <= 0:
        raise ValueError("signal_power must be positive")
    kwargs: dict[str, object] = {}
    if like is not None:
        kwargs = {"dtype": like.real.dtype, "device": like.device}
    snr = torch.as_tensor(snr_db, **kwargs)
    if not torch.isfinite(snr).all():
        raise ValueError("snr_db contains non-finite values")
    signal = torch.as_tensor(signal_power, **kwargs)
    return signal * torch.pow(10.0, -snr / 10.0)


def _broadcast_noise_power(
    noise_power: float | torch.Tensor,
    symbols: torch.Tensor,
) -> torch.Tensor:
    value = torch.as_tensor(
        noise_power,
        dtype=symbols.real.dtype,
        device=symbols.device,
    )
    if not torch.isfinite(value).all() or torch.any(value < 0):
        raise ValueError("noise_power must be finite and non-negative")
    while value.ndim < symbols.ndim:
        value = value.unsqueeze(-1)
    try:
        return torch.broadcast_to(value, symbols.shape)
    except RuntimeError as exc:
        raise ValueError("noise_power is not broadcastable to symbols") from exc


class NativeComplexAWGN:
    """Small PyTorch reference used only for CPU tests and Sionna cross-checks."""

    backend_name = "native_reference"

    def __init__(self, *, seed: int = 0) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.seed = seed
        self._generators: dict[str, torch.Generator] = {}

    def _generator(self, device: torch.device, noise_seed: int | None) -> torch.Generator:
        key = str(device)
        if noise_seed is not None:
            if noise_seed < 0:
                raise ValueError("noise_seed must be non-negative")
            return torch.Generator(device=device).manual_seed(noise_seed)
        if key not in self._generators:
            self._generators[key] = torch.Generator(device=device).manual_seed(self.seed)
        return self._generators[key]

    def transmit(
        self,
        symbols: torch.Tensor,
        noise_power: float | torch.Tensor,
        *,
        noise_seed: int | None = None,
    ) -> torch.Tensor:
        """Apply complex AWGN with variance ``N₀/2`` per real component."""

        if not symbols.is_complex():
            raise ValueError("symbols must be complex")
        expanded_power = _broadcast_noise_power(noise_power, symbols)
        generator = self._generator(symbols.device, noise_seed)
        real = torch.randn(
            symbols.shape,
            dtype=symbols.real.dtype,
            device=symbols.device,
            generator=generator,
        )
        imag = torch.randn(
            symbols.shape,
            dtype=symbols.real.dtype,
            device=symbols.device,
            generator=generator,
        )
        unit_noise = torch.complex(real, imag) / (2.0**0.5)
        return symbols + unit_noise * expanded_power.sqrt()


class SionnaComplexAWGN:
    """Lazy adapter around :class:`sionna.phy.channel.AWGN`."""

    backend_name = "sionna_phy"

    def __init__(self, *, device: str, seed: int) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        try:
            from sionna.phy import config
            from sionna.phy.channel import AWGN
        except ImportError as exc:
            raise RuntimeError(
                "Sionna PHY is required for the formal E5 channel; "
                "install requirements/sionna.txt in its dedicated environment"
            ) from exc
        config.device = device
        config.seed = seed
        self._config = config
        self._channel = AWGN(device=device)

    def transmit(
        self,
        symbols: torch.Tensor,
        noise_power: float | torch.Tensor,
        *,
        noise_seed: int | None = None,
    ) -> torch.Tensor:
        if not symbols.is_complex():
            raise ValueError("symbols must be complex")
        if noise_seed is not None:
            if noise_seed < 0:
                raise ValueError("noise_seed must be non-negative")
            self._config.seed = noise_seed
        received: torch.Tensor = self._channel(symbols, noise_power)
        return received


def build_awgn_channel(
    backend: str,
    *,
    device: str,
    seed: int,
) -> ComplexAWGNChannel:
    """Build an explicit formal or reference AWGN backend."""

    if backend == "sionna":
        return SionnaComplexAWGN(device=device, seed=seed)
    if backend == "native_reference":
        return NativeComplexAWGN(seed=seed)
    raise ValueError("channel backend must be sionna or native_reference")
