"""Motion- and reconstruction-level metrics for E2 sensitivity experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from av_semcom.data.landmarks import FaceDetection, FaceLandmarkBackend


@dataclass(frozen=True)
class MotionMetrics:
    """Errors in the 18-D source-relative mouth representation."""

    l1: float
    rmse: float
    velocity_l1: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ReconstructionMetrics:
    """Aligned face and detected-mouth reconstruction metrics."""

    face_mae: float
    psnr_db: float
    ssim: float
    mouth_mae: float | None
    mouth_nme: float | None
    landmark_coverage: float

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


def compute_motion_metrics(target: np.ndarray, candidate: np.ndarray) -> MotionMetrics:
    """Compute errors between two ``[T, 18]`` mouth-motion sequences."""

    if target.shape != candidate.shape or target.ndim != 2 or target.shape[1] != 18:
        raise ValueError("target and candidate must share shape [T, 18]")
    difference = candidate.astype(np.float64) - target.astype(np.float64)
    velocity_difference = np.diff(candidate, axis=0) - np.diff(target, axis=0)
    velocity_l1 = float(np.abs(velocity_difference).mean()) if velocity_difference.size else 0.0
    return MotionMetrics(
        l1=float(np.abs(difference).mean()),
        rmse=float(np.sqrt(np.square(difference).mean())),
        velocity_l1=velocity_l1,
    )


def compute_reconstruction_metrics(
    target_frames: np.ndarray,
    reconstructed_frames: np.ndarray,
    *,
    landmark_backend: FaceLandmarkBackend | None = None,
    target_detections: Sequence[FaceDetection | None] | None = None,
) -> ReconstructionMetrics:
    """Measure aligned face quality and optional detected-mouth geometry."""

    if (
        target_frames.shape != reconstructed_frames.shape
        or target_frames.ndim != 4
        or target_frames.shape[-1] != 3
    ):
        raise ValueError("target and reconstructed frames must share shape [T, H, W, 3]")
    if target_frames.dtype != np.uint8 or reconstructed_frames.dtype != np.uint8:
        raise ValueError("reconstruction metrics require uint8 RGB frames")

    if target_detections is not None and len(target_detections) != target_frames.shape[0]:
        raise ValueError("target_detections must contain one entry per frame")

    target_float = target_frames.astype(np.float32)
    reconstructed_float = reconstructed_frames.astype(np.float32)
    difference = reconstructed_float - target_float
    mean_squared_error = float(np.square(difference).mean())
    psnr = (
        float("inf")
        if mean_squared_error == 0
        else float(20 * np.log10(255.0 / np.sqrt(mean_squared_error)))
    )

    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise RuntimeError(
            "scikit-image is required for SSIM; install requirements/base.txt"
        ) from exc
    ssim_values = [
        structural_similarity(target, reconstructed, channel_axis=2, data_range=255)
        for target, reconstructed in zip(target_frames, reconstructed_frames, strict=True)
    ]

    mouth_mae_values: list[float] = []
    nme_values: list[float] = []
    if landmark_backend is not None:
        height, width = target_frames.shape[1:3]
        for frame_index, (target, reconstructed) in enumerate(
            zip(target_frames, reconstructed_frames, strict=True)
        ):
            target_detection = (
                target_detections[frame_index]
                if target_detections is not None
                else landmark_backend.detect(target)
            )
            reconstructed_detection = landmark_backend.detect(reconstructed)
            if target_detection is None or reconstructed_detection is None:
                continue
            target_xy = target_detection.mouth_landmarks[:, :2]
            reconstructed_xy = reconstructed_detection.mouth_landmarks[:, :2]
            minimum = target_xy.min(axis=0)
            maximum = target_xy.max(axis=0)
            diagonal = float(np.linalg.norm(maximum - minimum))
            if diagonal <= 1e-8:
                continue
            nme_values.append(
                float(np.linalg.norm(reconstructed_xy - target_xy, axis=1).mean() / diagonal)
            )

            padding = 0.2 * (maximum - minimum)
            padded_minimum = np.maximum(minimum - padding, 0)
            padded_maximum = np.minimum(maximum + padding, 1)
            x0 = int(np.floor(padded_minimum[0] * width))
            y0 = int(np.floor(padded_minimum[1] * height))
            x1 = int(np.ceil(padded_maximum[0] * width))
            y1 = int(np.ceil(padded_maximum[1] * height))
            if x1 <= x0 or y1 <= y0:
                continue
            mouth_mae_values.append(
                float(
                    np.abs(
                        reconstructed_float[frame_index, y0:y1, x0:x1]
                        - target_float[frame_index, y0:y1, x0:x1]
                    ).mean()
                )
            )

    detected = len(nme_values)
    return ReconstructionMetrics(
        face_mae=float(np.abs(difference).mean()),
        psnr_db=psnr,
        ssim=float(np.mean(ssim_values)),
        mouth_mae=float(np.mean(mouth_mae_values)) if mouth_mae_values else None,
        mouth_nme=float(np.mean(nme_values)) if nme_values else None,
        landmark_coverage=detected / target_frames.shape[0],
    )
