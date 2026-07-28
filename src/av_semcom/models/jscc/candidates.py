"""Portable motion candidates shared by Sionna and LivePortrait environments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from av_semcom.data.preprocessing import atomic_save_npz

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class JSCCCondition:
    """One frozen residual-reconstruction condition."""

    condition_id: str
    family: str
    channel_uses: int | None
    model_seed: int | None
    snr_db: float | None
    noise_seed: int | None

    def __post_init__(self) -> None:
        if not self.condition_id:
            raise ValueError("condition_id must be non-empty")
        if self.family not in {
            "prediction_only",
            "full_residual_oracle",
            "audio_prediction",
            "full_motion_oracle",
            "noiseless_autoencoder",
            "jscc_awgn",
        }:
            raise ValueError(f"unsupported JSCC condition family: {self.family}")
        channel_condition = self.family in {"noiseless_autoencoder", "jscc_awgn"}
        if channel_condition != (self.channel_uses is not None and self.model_seed is not None):
            raise ValueError("channel conditions require channel_uses and model_seed")
        if self.family == "jscc_awgn":
            if self.snr_db is None or self.noise_seed is None:
                raise ValueError("jscc_awgn requires snr_db and noise_seed")
        elif self.snr_db is not None or self.noise_seed is not None:
            raise ValueError("only jscc_awgn may specify SNR and a noise seed")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> JSCCCondition:
        return cls(
            condition_id=str(payload["condition_id"]),
            family=str(payload["family"]),
            channel_uses=(
                None if payload.get("channel_uses") is None else int(payload["channel_uses"])
            ),
            model_seed=(None if payload.get("model_seed") is None else int(payload["model_seed"])),
            snr_db=None if payload.get("snr_db") is None else float(payload["snr_db"]),
            noise_seed=(None if payload.get("noise_seed") is None else int(payload["noise_seed"])),
        )


@dataclass(frozen=True)
class JSCCCandidateBundle:
    """All frozen motion candidates for one GRID utterance."""

    sample_id: str
    split: str
    speaker_id: str
    experiment_fingerprint: str
    candidate_fingerprint: str
    conditions: tuple[JSCCCondition, ...]
    vectors: FloatArray
    valid_mask: BoolArray

    def __post_init__(self) -> None:
        if not self.sample_id or not self.experiment_fingerprint or not self.candidate_fingerprint:
            raise ValueError("bundle identity and fingerprints must be non-empty")
        if not self.conditions:
            raise ValueError("candidate bundle must contain at least one condition")
        ids = [condition.condition_id for condition in self.conditions]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate condition IDs must be unique")
        if (
            self.vectors.dtype != np.float32
            or self.vectors.ndim != 3
            or self.vectors.shape[0] != len(self.conditions)
            or self.vectors.shape[2] != 18
        ):
            raise ValueError("vectors must use float32 with shape [K,T,18]")
        if self.valid_mask.dtype != np.bool_ or self.valid_mask.shape != (self.vectors.shape[1],):
            raise ValueError("valid_mask must be boolean with shape [T]")
        if not np.isfinite(self.vectors).all():
            raise ValueError("candidate vectors contain non-finite values")
        if not np.allclose(self.vectors[:, 0], 0.0, atol=1e-7, rtol=0):
            raise ValueError("all candidate reference frames must be zero")

    def vector(self, condition_id: str) -> FloatArray:
        """Return one candidate by stable ID."""

        for index, condition in enumerate(self.conditions):
            if condition.condition_id == condition_id:
                return self.vectors[index].copy()
        raise KeyError(condition_id)


def save_candidate_bundle(path: Path, bundle: JSCCCandidateBundle) -> None:
    """Atomically save a portable no-pickle candidate bundle."""

    atomic_save_npz(
        path,
        sample_id=np.asarray(bundle.sample_id),
        split=np.asarray(bundle.split),
        speaker_id=np.asarray(bundle.speaker_id),
        experiment_fingerprint=np.asarray(bundle.experiment_fingerprint),
        candidate_fingerprint=np.asarray(bundle.candidate_fingerprint),
        conditions_json=np.asarray(
            json.dumps(
                [condition.to_dict() for condition in bundle.conditions],
                ensure_ascii=False,
                sort_keys=True,
            )
        ),
        vectors=bundle.vectors,
        valid_mask=bundle.valid_mask,
    )


def load_candidate_bundle(
    path: Path,
    *,
    expected_fingerprint: str,
) -> JSCCCandidateBundle:
    """Load and validate one candidate artifact."""

    with np.load(path, allow_pickle=False) as payload:
        fingerprint = str(payload["candidate_fingerprint"].item())
        if fingerprint != expected_fingerprint:
            raise ValueError(f"candidate fingerprint mismatch: {path}")
        conditions_payload = json.loads(str(payload["conditions_json"].item()))
        if not isinstance(conditions_payload, list):
            raise ValueError("conditions_json must contain a list")
        return JSCCCandidateBundle(
            sample_id=str(payload["sample_id"].item()),
            split=str(payload["split"].item()),
            speaker_id=str(payload["speaker_id"].item()),
            experiment_fingerprint=str(payload["experiment_fingerprint"].item()),
            candidate_fingerprint=fingerprint,
            conditions=tuple(
                JSCCCondition.from_dict(condition) for condition in conditions_payload
            ),
            vectors=payload["vectors"].astype(np.float32),
            valid_mask=payload["valid_mask"].astype(np.bool_),
        )


def condition_id(
    family: str,
    *,
    channel_uses: int | None = None,
    model_seed: int | None = None,
    snr_db: float | None = None,
    noise_seed: int | None = None,
) -> str:
    """Build a readable stable condition identifier."""

    if family in {
        "prediction_only",
        "full_residual_oracle",
        "audio_prediction",
        "full_motion_oracle",
    }:
        return family
    if channel_uses is None or model_seed is None:
        raise ValueError("channel condition IDs require channel_uses and model_seed")
    stem = f"{family}_c{channel_uses}_seed{model_seed}"
    if family == "jscc_awgn":
        if snr_db is None or noise_seed is None:
            raise ValueError("AWGN condition IDs require SNR and noise_seed")
        return f"{stem}_snr{snr_db:g}_noise{noise_seed}"
    return stem
