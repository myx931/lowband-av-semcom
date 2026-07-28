"""Face and mouth landmark extraction with an injectable backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from av_semcom.data.preprocessing import interpolate_missing

MOUTH_LANDMARK_INDICES: tuple[int, ...] = (
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
    185,
    40,
    39,
    37,
    0,
    267,
    269,
    270,
    409,
    415,
    310,
    311,
    312,
    13,
    82,
    81,
    80,
    191,
    78,
)


@dataclass(frozen=True)
class FaceDetection:
    """Normalized mouth landmarks and full-face bounding box."""

    mouth_landmarks: np.ndarray
    face_box: np.ndarray


class FaceLandmarkBackend(Protocol):
    """Minimal interface for replaceable landmark detectors."""

    def detect(self, rgb_image: np.ndarray) -> FaceDetection | None:
        """Detect one face in an RGB image."""

    def close(self) -> None:
        """Release backend resources."""


class MediaPipeFaceMeshBackend:
    """Pinned MediaPipe Face Mesh implementation."""

    def __init__(self) -> None:
        try:
            import mediapipe as mp
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "MediaPipe 0.10.21 is required for landmarks. Install requirements/base.txt."
            ) from exc
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, rgb_image: np.ndarray) -> FaceDetection | None:
        """Detect normalized landmarks and a full-face box."""

        result = self._mesh.process(rgb_image)
        if not result.multi_face_landmarks:
            return None
        all_points = np.asarray(
            [(point.x, point.y, point.z) for point in result.multi_face_landmarks[0].landmark],
            dtype=np.float32,
        )
        mouth = all_points[np.asarray(MOUTH_LANDMARK_INDICES)]
        minimum = all_points[:, :2].min(axis=0)
        maximum = all_points[:, :2].max(axis=0)
        face_box = np.asarray([minimum[0], minimum[1], maximum[0], maximum[1]], dtype=np.float32)
        return FaceDetection(mouth_landmarks=mouth, face_box=face_box)

    def close(self) -> None:
        """Release MediaPipe graph resources."""

        self._mesh.close()

    def reset(self) -> None:
        """Reset tracking state before an independent frame sequence."""

        self._mesh.reset()


def extract_landmark_sequence(
    frame_paths: list[Path],
    backend: FaceLandmarkBackend,
    *,
    min_detection_coverage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract, validate, and interpolate a landmark sequence.

    Returns:
        Interpolated mouth landmarks, interpolated face boxes, and the original
        per-frame detection mask.
    """

    if not frame_paths:
        raise ValueError("at least one frame is required")
    if not 0 < min_detection_coverage <= 1:
        raise ValueError("min_detection_coverage must be in (0, 1]")

    landmarks = np.full(
        (len(frame_paths), len(MOUTH_LANDMARK_INDICES), 3),
        np.nan,
        dtype=np.float32,
    )
    face_boxes = np.full((len(frame_paths), 4), np.nan, dtype=np.float32)
    valid_mask = np.zeros(len(frame_paths), dtype=np.bool_)
    for index, frame_path in enumerate(frame_paths):
        with Image.open(frame_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        detection = backend.detect(rgb)
        if detection is None:
            continue
        if detection.mouth_landmarks.shape != (len(MOUTH_LANDMARK_INDICES), 3):
            raise ValueError(
                "landmark backend returned an invalid mouth shape: "
                f"{detection.mouth_landmarks.shape}"
            )
        if detection.face_box.shape != (4,):
            raise ValueError(
                f"landmark backend returned an invalid face box: {detection.face_box.shape}"
            )
        landmarks[index] = detection.mouth_landmarks
        face_boxes[index] = detection.face_box
        valid_mask[index] = True

    coverage = float(valid_mask.mean())
    if coverage < min_detection_coverage:
        raise ValueError(
            f"face detection coverage {coverage:.3f} is below "
            f"the required {min_detection_coverage:.3f}"
        )
    return (
        interpolate_missing(landmarks, valid_mask),
        interpolate_missing(face_boxes, valid_mask),
        valid_mask,
    )
