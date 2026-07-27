"""CPU smoke test for the E4 motion-space residual pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from av_semcom.data.grid import GridSample, GridSettings
from av_semcom.data.landmarks import MOUTH_LANDMARK_INDICES, FaceDetection
from av_semcom.data.preprocessing import atomic_save_npz
from av_semcom.models.motion.config import MotionSettings
from av_semcom.models.motion.perturbations import MotionNormalizer, save_motion_normalizer
from av_semcom.models.motion.sequence import save_motion_sequence
from av_semcom.models.predictor.artifacts import save_prediction
from av_semcom.models.predictor.config import AudioMotionSettings
from av_semcom.models.reconstruction.backend import FakeReconstructionBackend
from av_semcom.models.residual.config import ResidualSettings
from av_semcom.models.residual.experiment import run_residual_analysis
from av_semcom.models.residual.reconstruction import run_residual_reconstruction


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
def test_residual_analysis_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    config = _config(tmp_path)
    data = GridSettings.from_config(config)
    predictor = AudioMotionSettings.from_config(config, data)
    motion = MotionSettings.from_config(config, data)
    settings = ResidualSettings.from_config(config)
    save_motion_normalizer(
        predictor.motion_stats_path,
        MotionNormalizer(
            mean=np.zeros(18, dtype=np.float32),
            std=np.linspace(0.5, 2.0, 18, dtype=np.float32),
            scope="train_stats",
        ),
    )
    e3_run = tmp_path / "e3"
    fingerprint = "synthetic-e3"
    samples = [
        _sample("s1_demo", "s1", "validation"),
        _sample("s2_demo", "s2", "test"),
    ]
    _write_e3_metadata(e3_run, fingerprint)
    for index, sample in enumerate(samples, start=1):
        crops = np.zeros((4, 256, 256, 3), dtype=np.uint8)
        for frame_index in range(4):
            crops[frame_index].fill(index * 10 + frame_index * 30)
        sequence = FakeReconstructionBackend().extract_motion(
            crops,
            np.ones(4, dtype=np.bool_),
            sample_id=sample.sample_id,
            fps=25,
            config_fingerprint="smoke",
        )
        save_motion_sequence(data_root / str(sample.motion_path), sequence)
        atomic_save_npz(
            data_root / str(sample.face_crop_path),
            crops=crops,
            valid_mask=np.ones(4, dtype=np.bool_),
        )
        target = sequence.lip_vector
        prediction = target * np.float32(0.6)
        prediction[0] = 0
        save_prediction(
            e3_run / "seed_43" / "predictions" / sample.split / f"{sample.sample_id}.npz",
            sample_id=sample.sample_id,
            method="audio_gru",
            split=sample.split,
            speaker_id=sample.speaker_id,
            prediction=prediction,
            target=target,
            valid_mask=np.ones(4, dtype=np.bool_),
            seed=43,
            experiment_fingerprint=fingerprint,
        )

    run_dir = tmp_path / "residual_run"
    _, summary = run_residual_analysis(
        settings,
        predictor,
        samples,
        e3_run,
        run_directory=run_dir,
    )
    complete_mtime = (run_dir / "analysis_complete.json").stat().st_mtime_ns
    _, resumed = run_residual_analysis(
        settings,
        predictor,
        samples,
        e3_run,
        run_directory=run_dir,
        resume=True,
    )

    assert summary["sample_count"] == 2
    assert summary["selection_result_count"] == 66
    assert resumed == summary
    assert len(list((run_dir / "samples").glob("*.json"))) == 2
    assert (run_dir / "analysis_complete.json").stat().st_mtime_ns == complete_mtime
    assert {group["split"] for group in summary["groups"]} == {"validation", "test"}
    reconstruction, failures = run_residual_reconstruction(
        settings,
        motion,
        samples,
        e3_run,
        run_dir,
        backend=FakeReconstructionBackend(),
        landmark_backend=_ConstantLandmarks(),
        save_representative_media=False,
    )
    assert failures == []
    assert reconstruction["result_count"] == 46
    assert (run_dir / "reconstruction/complete.json").is_file()


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "data": {
            "root": None,
            "raw_video_dir": "grid/raw/video",
            "raw_audio_dir": "grid/raw/audio",
            "manifest_path": "grid/manifest.jsonl",
            "failure_dir": "grid/failures",
            "processed_dir": "grid/processed",
            "speakers": ["s1", "s2"],
            "max_samples": 1,
            "fps": 25,
            "split_seed": 42,
            "validation_ratio": 0.5,
            "test_ratio": 0.5,
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
            "reconstruction_batch_size": 2,
            "stats_filename": "train_stats.json",
            "stats_split": "train",
            "stats_scope": "train_stats",
        },
        "model": {
            "output_dim": 18,
            "mel_bins": 80,
            "mel_steps_per_frame": 4,
            "audio_projection_dim": 8,
            "hidden_dim": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "bidirectional": False,
        },
        "training": {
            "seeds": [42, 43, 44],
            "device": "cpu",
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "max_epochs": 1,
            "early_stopping_patience": 1,
            "early_stopping_min_delta": 0.0,
            "gradient_clip_norm": 1.0,
            "num_workers": 0,
            "mixed_precision": False,
            "deterministic": True,
        },
        "evaluation": {
            "output_dir": str(tmp_path / "e3_outputs"),
            "splits": ["validation", "test"],
            "baselines": ["zero_motion", "train_mean", "oracle_persistence"],
        },
        "experiment": {
            "output_dir": str(tmp_path / "residual_outputs"),
            "metric_workers": 1,
        },
        "residual": {
            "selection_spaces": ["raw", "normalized"],
            "budgets": [0, 1, 2, 4, 6, 9, 12, 18],
            "random_seeds": [42, 43, 44],
            "reconstruction_budgets": [0, 2, 4, 6, 9, 18],
            "value_storage_bits": 32,
            "dimension_index_bits": 5,
        },
    }


def _sample(sample_id: str, speaker: str, split: str) -> GridSample:
    return GridSample(
        sample_id=sample_id,
        speaker_id=speaker,
        video_path=f"video/{sample_id}",
        audio_path=f"audio/{sample_id}.wav",
        fps=25,
        sample_rate=16000,
        frame_count=4,
        split=split,
        audio_feature_path=f"audio_features/{sample_id}.npz",
        face_crop_path=f"crops/{sample_id}.npz",
        motion_path=f"motion/{sample_id}.npz",
        status="processed",
    )


def _write_e3_metadata(run_dir: Path, fingerprint: str) -> None:
    (run_dir / "reconstruction").mkdir(parents=True)
    (run_dir / "experiment.json").write_text(
        json.dumps({"status": "complete", "experiment_fingerprint": fingerprint}),
        encoding="utf-8",
    )
    (run_dir / "validation_report.json").write_text(
        json.dumps({"error_count": 0}),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "groups": [
                    {"method": "audio_gru", "split": "validation", "seed": 42, "l1": 0.3},
                    {"method": "audio_gru", "split": "validation", "seed": 43, "l1": 0.2},
                    {"method": "audio_gru", "split": "validation", "seed": 44, "l1": 0.4},
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "reconstruction" / "complete.json").write_text(
        json.dumps(
            {
                "best_validation_seed": 43,
                "sample_count": 2,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "reconstruction" / "summary.json").write_text(
        json.dumps({"failure_count": 0}),
        encoding="utf-8",
    )
