"""Tests for injectable landmarks and face cropping."""

from pathlib import Path

import numpy as np
from PIL import Image

from av_semcom.data.face_crop import extract_face_crops
from av_semcom.data.landmarks import (
    MOUTH_LANDMARK_INDICES,
    FaceDetection,
    extract_landmark_sequence,
)


class _FakeBackend:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, rgb_image: np.ndarray) -> FaceDetection | None:
        del rgb_image
        self.calls += 1
        if self.calls == 2:
            return None
        landmarks = np.full((len(MOUTH_LANDMARK_INDICES), 3), self.calls / 10, np.float32)
        return FaceDetection(
            mouth_landmarks=landmarks,
            face_box=np.asarray([0.2, 0.2, 0.8, 0.8], dtype=np.float32),
        )

    def close(self) -> None:
        return None


def test_landmark_interpolation_and_face_crop(tmp_path: Path) -> None:
    frame_paths: list[Path] = []
    for index in range(3):
        path = tmp_path / f"{index:03d}.jpg"
        Image.new("RGB", (40, 30), color=(index * 30, 20, 10)).save(path)
        frame_paths.append(path)

    landmarks, boxes, valid = extract_landmark_sequence(
        frame_paths,
        _FakeBackend(),
        min_detection_coverage=0.6,
    )
    crops = extract_face_crops(frame_paths, boxes, image_size=16, padding=0.2)

    assert landmarks.shape == (3, 40, 3)
    assert boxes.shape == (3, 4)
    assert np.array_equal(valid, np.asarray([True, False, True]))
    assert np.isfinite(landmarks).all()
    assert crops.shape == (3, 16, 16, 3)
    assert crops.dtype == np.uint8
