from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from av_semcom.channel.awgn import NativeComplexAWGN
from av_semcom.data.grid import GridSample, GridSettings
from av_semcom.data.landmarks import MOUTH_LANDMARK_INDICES, FaceDetection
from av_semcom.data.preprocessing import atomic_save_npz, atomic_write_json
from av_semcom.models.jscc.candidates import save_candidate_bundle
from av_semcom.models.jscc.config import JSCCReconstructionSettings, JSCCSettings
from av_semcom.models.jscc.data import ResidualExample
from av_semcom.models.jscc.export import _build_bundle
from av_semcom.models.jscc.model import ResidualJSCC
from av_semcom.models.jscc.reconstruction import run_jscc_reconstruction
from av_semcom.models.motion.config import MotionSettings
from av_semcom.models.motion.sequence import save_motion_sequence
from av_semcom.models.reconstruction.backend import FakeReconstructionBackend

pytestmark = pytest.mark.smoke


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


def test_jscc_candidate_to_reconstruction_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    config = _config(tmp_path)
    data = GridSettings.from_config(config)
    jscc = JSCCSettings.from_config(config)
    reconstruction = JSCCReconstructionSettings.from_config(config, jscc)
    motion = MotionSettings.from_config(config, data)
    sample = _sample()
    crops = np.zeros((5, 256, 256, 3), dtype=np.uint8)
    for index in range(5):
        crops[index].fill(20 + index * 30)
    fake = FakeReconstructionBackend()
    sequence = fake.extract_motion(
        crops,
        np.ones(5, dtype=np.bool_),
        sample_id=sample.sample_id,
        fps=25,
        config_fingerprint="smoke",
    )
    save_motion_sequence(data_root / str(sample.motion_path), sequence)
    atomic_save_npz(
        data_root / str(sample.face_crop_path),
        crops=crops,
        valid_mask=np.ones(5, dtype=np.bool_),
    )
    target = sequence.lip_vector.astype(np.float32)
    prediction = target * np.float32(0.5)
    prediction[0] = 0
    raw = target - prediction
    transmission = np.ones(5, dtype=np.bool_)
    transmission[0] = False
    example = ResidualExample(
        sample_id=sample.sample_id,
        speaker_id=sample.speaker_id,
        split=sample.split,
        prediction=prediction,
        target=target,
        raw_residual=raw,
        normalized_residual=raw.copy(),
        valid_mask=np.ones(5, dtype=np.bool_),
        transmission_mask=transmission,
    )
    model = ResidualJSCC(
        channel=NativeComplexAWGN(seed=42),
        hidden_dim=8,
        channel_uses=1,
    )
    bundle = _build_bundle(
        example,
        0,
        {1: model},
        {1: 42},
        jscc,
        reconstruction,
        np.ones(18, dtype=np.float32),
        "experiment",
        "candidate",
    )
    run_dir = tmp_path / "run"
    save_candidate_bundle(
        run_dir / "reconstruction_candidates/test/s7_demo.npz",
        bundle,
    )
    atomic_write_json(
        run_dir / "reconstruction_candidates/complete.json",
        {
            "status": "complete",
            "experiment_fingerprint": "experiment",
            "candidate_fingerprint": "candidate",
            "sample_count": 1,
            "condition_count_per_sample": 4,
            "noise_seed": 42,
        },
    )

    summary, failures = run_jscc_reconstruction(
        jscc,
        reconstruction,
        motion,
        [sample],
        run_dir,
        backend=fake,
        landmark_backend=_ConstantLandmarks(),
    )
    complete_mtime = (run_dir / "video_reconstruction/complete.json").stat().st_mtime_ns
    resumed, resumed_failures = run_jscc_reconstruction(
        jscc,
        reconstruction,
        motion,
        [sample],
        run_dir,
        resume=True,
        backend=fake,
        landmark_backend=_ConstantLandmarks(),
    )

    assert failures == resumed_failures == []
    assert summary["result_count"] == 4
    assert len(summary["groups"]) == 4
    assert resumed == summary
    assert (run_dir / "video_reconstruction/complete.json").stat().st_mtime_ns == complete_mtime


def _sample() -> GridSample:
    return GridSample(
        sample_id="s7_demo",
        speaker_id="s7",
        video_path="video/s7_demo",
        audio_path="audio/s7_demo.wav",
        fps=25,
        sample_rate=16000,
        frame_count=5,
        split="test",
        audio_feature_path="audio_features/s7_demo.npz",
        face_crop_path="crops/s7_demo.npz",
        motion_path="motion/s7_demo.npz",
        status="processed",
    )


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "data": {
            "root": None,
            "raw_video_dir": "grid/raw/video",
            "raw_audio_dir": "grid/raw/audio",
            "manifest_path": "grid/manifest.jsonl",
            "failure_dir": "grid/failures",
            "processed_dir": "grid/processed",
            "speakers": ["s7"],
            "max_samples": 1,
            "fps": 25,
            "split_seed": 42,
            "validation_ratio": 0.1,
            "test_ratio": 0.1,
        },
        "motion": {
            "backend": "fake",
            "backend_revision": "test-v1",
            "repository": "third_party/LivePortrait",
            "model_root_env": "MODEL_ROOT",
            "output_dir": "grid/motion",
            "device": "cpu",
            "half_precision": False,
            "stitching": True,
            "reconstruction_batch_size": 4,
            "stats_filename": "train_stats.json",
            "stats_split": "train",
            "stats_scope": "train_stats",
        },
        "channel": {
            "backend": "native_reference",
            "complex_channel_uses": [1],
            "target_power": 1.0,
        },
        "jscc_model": {"input_dim": 18, "hidden_dim": 8},
        "jscc_training": {
            "seeds": [42],
            "device": "cpu",
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "max_epochs": 1,
            "early_stopping_patience": 1,
            "early_stopping_min_delta": 0.0,
            "gradient_clip_norm": 1.0,
            "num_workers": 0,
            "deterministic": True,
            "snr_min_db": 0.0,
            "snr_max_db": 5.0,
        },
        "jscc_evaluation": {
            "output_dir": str(tmp_path / "outputs"),
            "validation_snr_db": [2.5],
            "test_snr_db": [0.0],
            "noise_seeds": [42],
        },
        "jscc_reconstruction": {
            "split": "test",
            "noise_seed": 42,
            "metric_workers": 1,
            "media_channel_uses": 1,
            "save_representative_media": False,
        },
    }
