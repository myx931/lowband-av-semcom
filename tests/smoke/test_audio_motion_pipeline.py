"""CPU end-to-end smoke test for E3 training and prediction artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from av_semcom.data.grid import GridSample, GridSettings, write_manifest
from av_semcom.data.landmarks import MOUTH_LANDMARK_INDICES, FaceDetection
from av_semcom.data.preprocessing import atomic_save_npz
from av_semcom.models.motion.config import MotionSettings
from av_semcom.models.motion.perturbations import (
    fit_motion_normalizer,
    save_motion_normalizer,
)
from av_semcom.models.motion.sequence import save_motion_sequence
from av_semcom.models.predictor.config import AudioMotionSettings
from av_semcom.models.predictor.experiment import run_audio_motion_experiment
from av_semcom.models.predictor.reconstruction import run_prediction_reconstruction
from av_semcom.models.reconstruction.backend import FakeReconstructionBackend


def _sample(root: Path, speaker: str, split: str, value: int) -> tuple[GridSample, object]:
    sample_id = f"{speaker}_{split}"
    frame_count = 5
    audio_path = root / f"grid/processed/audio/{speaker}/{sample_id}.npz"
    rng = np.random.default_rng(value)
    atomic_save_npz(
        audio_path,
        features=rng.normal(size=(frame_count, 4, 80)).astype(np.float32),
    )
    crops = np.zeros((frame_count, 256, 256, 3), dtype=np.uint8)
    for index in range(frame_count):
        crops[index].fill(value + index * 20)
    sequence = FakeReconstructionBackend().extract_motion(
        crops,
        np.ones(frame_count, dtype=np.bool_),
        sample_id=sample_id,
        fps=25,
        config_fingerprint="smoke",
    )
    motion_path = root / f"grid/processed/motion/{speaker}/{sample_id}.npz"
    save_motion_sequence(motion_path, sequence)
    crop_path = root / f"grid/processed/crops/{speaker}/{sample_id}.npz"
    atomic_save_npz(
        crop_path,
        crops=crops,
        valid_mask=np.ones(frame_count, dtype=np.bool_),
    )
    sample = GridSample(
        sample_id=sample_id,
        speaker_id=speaker,
        video_path=f"grid/raw/video/{speaker}/{sample_id}",
        audio_path=f"grid/raw/audio/{speaker}/{sample_id}.wav",
        fps=25,
        sample_rate=25000,
        frame_count=frame_count,
        split=split,
        audio_feature_path=f"grid/processed/audio/{speaker}/{sample_id}.npz",
        face_crop_path=f"grid/processed/crops/{speaker}/{sample_id}.npz",
        motion_path=f"grid/processed/motion/{speaker}/{sample_id}.npz",
        status="processed",
    )
    return sample, sequence


class _ConstantLandmarks:
    def detect(self, rgb_image: np.ndarray) -> FaceDetection:
        del rgb_image
        x = np.linspace(0.4, 0.6, len(MOUTH_LANDMARK_INDICES), dtype=np.float32)
        points = np.stack([x, np.full_like(x, 0.7), np.zeros_like(x)], axis=1)
        return FaceDetection(
            mouth_landmarks=points,
            face_box=np.asarray([0.1, 0.1, 0.9, 0.9], dtype=np.float32),
        )

    def close(self) -> None:
        return None


@pytest.mark.smoke
def test_audio_motion_training_smoke(tmp_path: Path) -> None:
    train, train_sequence = _sample(tmp_path, "s3", "train", 10)
    validation, _ = _sample(tmp_path, "s1", "validation", 20)
    test, _ = _sample(tmp_path, "s2", "test", 30)
    stats_path = tmp_path / "grid/processed/motion/train_stats.json"
    save_motion_normalizer(
        stats_path,
        fit_motion_normalizer([train_sequence], scope="train_stats"),
    )
    config = {
        "data": {
            "root": str(tmp_path),
            "raw_video_dir": "grid/raw/video",
            "raw_audio_dir": "grid/raw/audio",
            "manifest_path": "grid/manifests/data.jsonl",
            "failure_dir": "grid/manifests/failures",
            "processed_dir": "grid/processed",
            "speakers": ["s1", "s2", "s3"],
            "max_samples": 1,
            "fps": 25,
            "split_seed": 42,
            "validation_ratio": 0.1,
            "test_ratio": 0.1,
            "resume": True,
        },
        "motion": {
            "backend": "fake",
            "backend_revision": "test-v1",
            "output_dir": "grid/processed/motion",
            "stats_filename": "train_stats.json",
        },
        "model": {
            "mel_bins": 80,
            "mel_steps_per_frame": 4,
            "audio_projection_dim": 8,
            "hidden_dim": 12,
            "num_layers": 1,
            "dropout": 0,
            "output_dim": 18,
            "bidirectional": False,
        },
        "training": {
            "seeds": [42],
            "device": "cpu",
            "batch_size": 1,
            "learning_rate": 0.001,
            "weight_decay": 0,
            "max_epochs": 2,
            "early_stopping_patience": 1,
            "early_stopping_min_delta": 0,
            "gradient_clip_norm": 1,
            "num_workers": 0,
            "mixed_precision": False,
            "deterministic": True,
        },
        "evaluation": {
            "output_dir": str(tmp_path / "outputs"),
            "splits": ["validation", "test"],
            "baselines": ["zero_motion", "train_mean", "oracle_persistence"],
        },
        "experiment": {
            "output_dir": str(tmp_path / "outputs/reconstruction"),
            "save_sample_positions": [],
        },
    }
    data_settings = GridSettings.from_config(config)
    settings = AudioMotionSettings.from_config(config, data_settings)
    write_manifest(data_settings.manifest_path, [train, validation, test])

    run_dir, summary = run_audio_motion_experiment(
        settings,
        [train, validation, test],
    )

    assert summary["result_count"] == 8
    assert (run_dir / "seed_42/best.pt").is_file()
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "predictions/zero_motion/test/s2_test.npz").is_file()
    assert (run_dir / "seed_42/predictions/test/s2_test.npz").is_file()

    motion_settings = MotionSettings.from_config(config, data_settings)
    reconstruction_summary, failures = run_prediction_reconstruction(
        settings,
        motion_settings,
        [train, validation, test],
        run_dir,
        backend=FakeReconstructionBackend(),
        landmark_backend=_ConstantLandmarks(),
        save_representative_media=False,
    )
    assert failures == []
    assert reconstruction_summary["result_count"] == 8
