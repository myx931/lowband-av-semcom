"""Typed motion sequences shared by extraction and reconstruction backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from av_semcom.data.preprocessing import atomic_save_npz

LIVEPORTRAIT_LIP_INDICES: tuple[int, ...] = (6, 12, 14, 17, 19, 20)


@dataclass(frozen=True)
class MotionSequence:
    """A portable LivePortrait-compatible motion sequence.

    Expression and canonical-keypoint tensors retain the full 21-point
    representation needed by the frozen renderer. ``lip_delta`` isolates the
    six mouth-related expression points relative to the selected source frame.
    """

    sample_id: str
    fps: float
    backend: str
    backend_revision: str
    config_fingerprint: str
    source_frame_index: int
    expression: np.ndarray
    lip_delta: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    scale: np.ndarray
    canonical_keypoints: np.ndarray
    valid_mask: np.ndarray
    lip_indices: np.ndarray

    def __post_init__(self) -> None:
        """Reject malformed artifacts before they reach an experiment."""

        frame_count = self.expression.shape[0]
        expected_shapes = {
            "expression": (frame_count, 21, 3),
            "lip_delta": (frame_count, 6, 3),
            "rotation": (frame_count, 3, 3),
            "translation": (frame_count, 3),
            "scale": (frame_count, 1),
            "canonical_keypoints": (frame_count, 21, 3),
            "valid_mask": (frame_count,),
            "lip_indices": (6,),
        }
        arrays = {
            "expression": self.expression,
            "lip_delta": self.lip_delta,
            "rotation": self.rotation,
            "translation": self.translation,
            "scale": self.scale,
            "canonical_keypoints": self.canonical_keypoints,
            "valid_mask": self.valid_mask,
            "lip_indices": self.lip_indices,
        }
        if frame_count <= 0:
            raise ValueError("motion sequence must contain at least one frame")
        for name, expected in expected_shapes.items():
            if arrays[name].shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {arrays[name].shape}")
        if not 0 <= self.source_frame_index < frame_count:
            raise ValueError("source_frame_index is outside the sequence")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if not np.array_equal(
            self.lip_indices.astype(np.int64),
            np.asarray(LIVEPORTRAIT_LIP_INDICES, dtype=np.int64),
        ):
            raise ValueError("lip_indices do not match the pinned LivePortrait mouth mapping")
        if not np.allclose(self.lip_delta[self.source_frame_index], 0.0, atol=1e-6):
            raise ValueError("lip_delta must be zero at source_frame_index")
        for name in (
            "expression",
            "lip_delta",
            "rotation",
            "translation",
            "scale",
            "canonical_keypoints",
        ):
            if not np.isfinite(arrays[name]).all():
                raise ValueError(f"{name} contains non-finite values")

    @property
    def frame_count(self) -> int:
        """Number of time steps in the sequence."""

        return int(self.expression.shape[0])

    @property
    def lip_vector(self) -> np.ndarray:
        """Return mouth motion as a contiguous ``[T, 18]`` array."""

        return np.ascontiguousarray(self.lip_delta.reshape(self.frame_count, 18))

    def with_lip_vector(self, lip_vector: np.ndarray) -> MotionSequence:
        """Return a copy whose mouth motion is replaced for reconstruction."""

        if lip_vector.shape != (self.frame_count, 18):
            raise ValueError(
                f"lip_vector must have shape {(self.frame_count, 18)}, got {lip_vector.shape}"
            )
        return MotionSequence(
            sample_id=self.sample_id,
            fps=self.fps,
            backend=self.backend,
            backend_revision=self.backend_revision,
            config_fingerprint=self.config_fingerprint,
            source_frame_index=self.source_frame_index,
            expression=self.expression,
            lip_delta=lip_vector.reshape(self.frame_count, 6, 3).astype(np.float32, copy=False),
            rotation=self.rotation,
            translation=self.translation,
            scale=self.scale,
            canonical_keypoints=self.canonical_keypoints,
            valid_mask=self.valid_mask,
            lip_indices=self.lip_indices,
        )


def save_motion_sequence(path: Path, sequence: MotionSequence) -> None:
    """Atomically save a motion sequence without pickle."""

    atomic_save_npz(
        path,
        sample_id=np.asarray(sequence.sample_id),
        fps=np.asarray(sequence.fps, dtype=np.float32),
        backend=np.asarray(sequence.backend),
        backend_revision=np.asarray(sequence.backend_revision),
        config_fingerprint=np.asarray(sequence.config_fingerprint),
        source_frame_index=np.asarray(sequence.source_frame_index, dtype=np.int64),
        expression=sequence.expression.astype(np.float32, copy=False),
        lip_delta=sequence.lip_delta.astype(np.float32, copy=False),
        rotation=sequence.rotation.astype(np.float32, copy=False),
        translation=sequence.translation.astype(np.float32, copy=False),
        scale=sequence.scale.astype(np.float32, copy=False),
        canonical_keypoints=sequence.canonical_keypoints.astype(np.float32, copy=False),
        valid_mask=sequence.valid_mask.astype(np.bool_, copy=False),
        lip_indices=sequence.lip_indices.astype(np.int64, copy=False),
    )


def load_motion_sequence(path: Path) -> MotionSequence:
    """Load and validate a motion sequence."""

    with np.load(path, allow_pickle=False) as data:
        return MotionSequence(
            sample_id=str(data["sample_id"].item()),
            fps=float(data["fps"].item()),
            backend=str(data["backend"].item()),
            backend_revision=str(data["backend_revision"].item()),
            config_fingerprint=str(data["config_fingerprint"].item()),
            source_frame_index=int(data["source_frame_index"].item()),
            expression=data["expression"].astype(np.float32),
            lip_delta=data["lip_delta"].astype(np.float32),
            rotation=data["rotation"].astype(np.float32),
            translation=data["translation"].astype(np.float32),
            scale=data["scale"].astype(np.float32),
            canonical_keypoints=data["canonical_keypoints"].astype(np.float32),
            valid_mask=data["valid_mask"].astype(np.bool_),
            lip_indices=data["lip_indices"].astype(np.int64),
        )
