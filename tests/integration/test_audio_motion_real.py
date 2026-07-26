"""Opt-in real CUDA batch and LivePortrait reconstruction test for E3."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from av_semcom.data.grid import GridSettings, read_manifest, resolve_record_path
from av_semcom.models.motion.config import MotionSettings
from av_semcom.models.motion.perturbations import load_motion_normalizer
from av_semcom.models.motion.pipeline import create_reconstruction_backend
from av_semcom.models.motion.sequence import load_motion_sequence
from av_semcom.models.predictor.config import AudioMotionSettings
from av_semcom.models.predictor.data import (
    AudioMotionDataset,
    fit_audio_normalizer,
    select_predictor_samples,
)
from av_semcom.models.predictor.model import AudioToMotionGRU, masked_l1_loss
from av_semcom.utils.config import load_yaml_config


@pytest.mark.integration
def test_real_audio_motion_cuda_and_reconstruction() -> None:
    if os.environ.get("RUN_AUDIO_MOTION_INTEGRATION") != "1":
        pytest.skip("set RUN_AUDIO_MOTION_INTEGRATION=1 in the GPU environment")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    config = load_yaml_config(Path("configs/experiment/audio_to_motion_gru.yaml"))
    data_settings = GridSettings.from_config(config)
    settings = AudioMotionSettings.from_config(config, data_settings)
    motion_settings = MotionSettings.from_config(config, data_settings)
    samples = select_predictor_samples(
        read_manifest(data_settings.manifest_path),
        data_settings,
    )
    train_samples = [sample for sample in samples if sample.split == "train"]
    audio_normalizer = fit_audio_normalizer(train_samples, data_settings.data_root)
    motion_normalizer = load_motion_normalizer(settings.motion_stats_path)
    dataset = AudioMotionDataset(
        train_samples[:1],
        data_settings.data_root,
        audio_normalizer,
        motion_normalizer,
    )
    item = dataset[0]
    audio = item["audio"].unsqueeze(0).cuda()
    target = item["target"].unsqueeze(0).cuda()
    mask = item["mask"].unsqueeze(0).cuda()
    model = AudioToMotionGRU(
        mel_bins=settings.mel_bins,
        mel_steps_per_frame=settings.mel_steps_per_frame,
        audio_projection_dim=settings.audio_projection_dim,
        hidden_dim=settings.hidden_dim,
        num_layers=settings.num_layers,
        dropout=settings.dropout,
        output_dim=settings.output_dim,
    ).cuda()
    prediction = model(audio)
    loss = masked_l1_loss(prediction, target, mask)
    loss.backward()
    assert prediction.shape == (1, 75, 18)
    assert np.isfinite(float(loss.detach().cpu()))

    sample = train_samples[0]
    sequence = load_motion_sequence(
        resolve_record_path(sample.motion_path, data_settings.data_root)
    )
    with np.load(
        resolve_record_path(sample.face_crop_path, data_settings.data_root),
        allow_pickle=False,
    ) as payload:
        crops = payload["crops"].astype(np.uint8)
    backend = create_reconstruction_backend(motion_settings)
    try:
        frames = backend.reconstruct(
            crops[sequence.source_frame_index],
            sequence,
            mode="lip_only",
            lip_vector=np.zeros((75, 18), dtype=np.float32),
        )
    finally:
        backend.close()
    assert frames.shape == (75, 256, 256, 3)
