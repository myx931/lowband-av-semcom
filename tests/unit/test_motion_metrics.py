"""Tests for motion and reconstruction metrics."""

from __future__ import annotations

import numpy as np

from av_semcom.data.landmarks import MOUTH_LANDMARK_INDICES, FaceDetection
from av_semcom.metrics.motion import (
    compute_motion_metrics,
    compute_reconstruction_metrics,
)


class _ConstantLandmarks:
    def detect(self, rgb_image: np.ndarray) -> FaceDetection:
        del rgb_image
        x = np.linspace(0.4, 0.6, len(MOUTH_LANDMARK_INDICES), dtype=np.float32)
        landmarks = np.stack([x, np.full_like(x, 0.7), np.zeros_like(x)], axis=1)
        return FaceDetection(
            mouth_landmarks=landmarks,
            face_box=np.asarray([0.1, 0.1, 0.9, 0.9], dtype=np.float32),
        )

    def close(self) -> None:
        return None


def test_identical_motion_and_frames_have_zero_error() -> None:
    motion = np.zeros((3, 18), dtype=np.float32)
    frames = np.full((3, 32, 32, 3), 120, dtype=np.uint8)

    motion_metrics = compute_motion_metrics(motion, motion)
    reconstruction_metrics = compute_reconstruction_metrics(
        frames,
        frames,
        landmark_backend=_ConstantLandmarks(),
    )

    assert motion_metrics.l1 == 0
    assert motion_metrics.rmse == 0
    assert motion_metrics.velocity_l1 == 0
    assert reconstruction_metrics.face_mae == 0
    assert reconstruction_metrics.psnr_db == float("inf")
    assert reconstruction_metrics.ssim == 1
    assert reconstruction_metrics.mouth_mae == 0
    assert reconstruction_metrics.mouth_nme == 0
    assert reconstruction_metrics.landmark_coverage == 1
