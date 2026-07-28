"""Channel-aware residual selection and safety baselines."""

from av_semcom.models.selection.config import (
    ChannelGateSettings,
    ResidualScorerSettings,
)
from av_semcom.models.selection.gate import GatePolicy, run_channel_gate_experiment
from av_semcom.models.selection.scorer import ChannelAwareResidualScorer
from av_semcom.models.selection.scorer_experiment import (
    run_scorer_evaluation,
    run_scorer_training,
)

__all__ = [
    "ChannelAwareResidualScorer",
    "ChannelGateSettings",
    "GatePolicy",
    "ResidualScorerSettings",
    "run_channel_gate_experiment",
    "run_scorer_evaluation",
    "run_scorer_training",
]
