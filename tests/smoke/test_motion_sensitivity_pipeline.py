"""CPU-only end-to-end smoke test for the E2 sensitivity pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from av_semcom.data.grid import GridSample, GridSettings, write_manifest
from av_semcom.data.landmarks import MOUTH_LANDMARK_INDICES, FaceDetection
from av_semcom.data.preprocessing import atomic_save_npz
from av_semcom.models.motion.config import MotionSettings
from av_semcom.models.motion.experiment import run_motion_sensitivity
from av_semcom.models.motion.perturbations import PerturbationCondition
from av_semcom.models.motion.pipeline import extract_motion_for_manifest
from av_semcom.models.reconstruction.backend import FakeReconstructionBackend


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


def _settings(root: Path) -> MotionSettings:
    config = {
        "data": {
            "root": str(root),
            "raw_video_dir": "grid/raw/video",
            "raw_audio_dir": "grid/raw/audio",
            "manifest_path": "grid/manifests/subset.jsonl",
            "failure_dir": "grid/manifests/failures",
            "processed_dir": "grid/processed",
            "speakers": ["s1"],
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
            "output_dir": "grid/processed/motion/fake",
        },
        "experiment": {
            "output_dir": str(root / "outputs"),
            "save_sample_positions": [],
        },
    }
    data_settings = GridSettings.from_config(config)
    return MotionSettings.from_config(config, data_settings)


@pytest.mark.smoke
def test_fake_motion_sensitivity_pipeline(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    crops = np.zeros((3, 256, 256, 3), dtype=np.uint8)
    crops[1].fill(40)
    crops[2].fill(80)
    crop_path = tmp_path / "grid/processed/face_crops/s1/s1_example.npz"
    atomic_save_npz(
        crop_path,
        crops=crops,
        valid_mask=np.ones(3, dtype=np.bool_),
    )
    sample = GridSample(
        sample_id="s1_example",
        speaker_id="s1",
        video_path="grid/raw/video/s1/example",
        audio_path="grid/raw/audio/s1/example.wav",
        fps=25,
        sample_rate=25000,
        frame_count=3,
        split="pilot",
        face_crop_path="grid/processed/face_crops/s1/s1_example.npz",
        status="processed",
    )
    write_manifest(settings.data_settings.manifest_path, [sample])
    updated, failures, _ = extract_motion_for_manifest(
        settings,
        [sample],
        backend=FakeReconstructionBackend(),
    )
    assert failures == []
    write_manifest(settings.data_settings.manifest_path, updated)

    run_directory, summary, run_failures = run_motion_sensitivity(
        settings,
        backend=FakeReconstructionBackend(),
        landmark_backend=_ConstantLandmarks(),
        output_directory=tmp_path / "run",
        conditions=(
            PerturbationCondition("lip_only", "identity"),
            PerturbationCondition("frozen", "frozen"),
        ),
    )

    assert run_failures == []
    assert summary["result_count"] == 3
    assert summary["condition_count"] == 3
    assert (run_directory / "results.jsonl").is_file()
    assert (run_directory / "summary.json").is_file()
    assert (run_directory / "environment.json").is_file()
