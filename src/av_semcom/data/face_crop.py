"""Face cropping driven by normalized landmark bounding boxes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def extract_face_crops(
    frame_paths: list[Path],
    face_boxes: np.ndarray,
    *,
    image_size: int,
    padding: float,
) -> np.ndarray:
    """Crop and resize RGB faces into a ``[T, H, W, 3]`` uint8 array."""

    if len(frame_paths) != face_boxes.shape[0] or face_boxes.shape[1:] != (4,):
        raise ValueError("face_boxes must have shape [len(frame_paths), 4]")
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if padding < 0:
        raise ValueError("padding must be non-negative")

    crops: list[np.ndarray] = []
    for frame_path, box in zip(frame_paths, face_boxes, strict=True):
        with Image.open(frame_path) as source:
            image = source.convert("RGB")
            width, height = image.size
            left, top, right, bottom = (float(value) for value in box)
            box_width = max(right - left, 1 / width)
            box_height = max(bottom - top, 1 / height)
            left = max(0.0, left - box_width * padding)
            right = min(1.0, right + box_width * padding)
            top = max(0.0, top - box_height * padding)
            bottom = min(1.0, bottom + box_height * padding)
            pixel_box = (
                round(left * width),
                round(top * height),
                round(right * width),
                round(bottom * height),
            )
            if pixel_box[2] <= pixel_box[0] or pixel_box[3] <= pixel_box[1]:
                raise ValueError(f"invalid face box for {frame_path}: {pixel_box}")
            crop = image.crop(pixel_box).resize(
                (image_size, image_size),
                resample=Image.Resampling.LANCZOS,
            )
            crops.append(np.asarray(crop, dtype=np.uint8))
    return np.stack(crops)
