"""Opt-in real LivePortrait extraction and 75-frame reconstruction test."""

from __future__ import annotations

import os

import numpy as np
import pytest

from av_semcom.data.grid import GridSettings, read_manifest, resolve_record_path
from av_semcom.models.motion.config import MotionSettings
from av_semcom.models.motion.pipeline import (
    create_reconstruction_backend,
    motion_artifact_fingerprint,
)
from av_semcom.utils.config import load_yaml_config


@pytest.mark.integration
def test_real_liveportrait_one_grid_sample() -> None:
    if os.environ.get("RUN_LIVEPORTRAIT_INTEGRATION") != "1":
        pytest.skip("set RUN_LIVEPORTRAIT_INTEGRATION=1 in the GPU environment")
    config = load_yaml_config("configs/motion/liveportrait.yaml")
    data_settings = GridSettings.from_config(config)
    motion_settings = MotionSettings.from_config(config, data_settings)
    samples = read_manifest(data_settings.manifest_path)
    if not samples or samples[0].face_crop_path is None:
        pytest.skip("processed GRID face crops are unavailable")

    sample = samples[0]
    crop_path = resolve_record_path(sample.face_crop_path, data_settings.data_root)
    with np.load(crop_path, allow_pickle=False) as data:
        crops = data["crops"].astype(np.uint8)
        valid_mask = data["valid_mask"].astype(np.bool_)
    backend = create_reconstruction_backend(motion_settings)
    try:
        sequence = backend.extract_motion(
            crops,
            valid_mask,
            sample_id=sample.sample_id,
            fps=sample.fps,
            config_fingerprint=motion_artifact_fingerprint(
                motion_settings,
                sample,
            ),
        )
        reconstructed = backend.reconstruct(
            crops[sequence.source_frame_index],
            sequence,
            mode="lip_only",
        )
    finally:
        backend.close()

    assert sequence.expression.shape == (75, 21, 3)
    assert sequence.lip_delta.shape == (75, 6, 3)
    assert reconstructed.shape == (75, 256, 256, 3)
    assert reconstructed.dtype == np.uint8
