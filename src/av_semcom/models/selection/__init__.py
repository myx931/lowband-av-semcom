"""Channel-aware residual selection and safety baselines."""

from av_semcom.models.selection.config import ChannelGateSettings
from av_semcom.models.selection.gate import GatePolicy, run_channel_gate_experiment

__all__ = [
    "ChannelGateSettings",
    "GatePolicy",
    "run_channel_gate_experiment",
]
