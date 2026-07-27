"""Unit tests for E3 data, causal model, baselines, and artifacts."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from av_semcom.data.grid import GridSample, GridSettings
from av_semcom.data.preprocessing import atomic_save_npz
from av_semcom.models.motion.perturbations import fit_motion_normalizer
from av_semcom.models.motion.sequence import save_motion_sequence
from av_semcom.models.predictor import reconstruction
from av_semcom.models.predictor.artifacts import (
    atomic_save_checkpoint,
    load_checkpoint,
    load_prediction,
    save_prediction,
)
from av_semcom.models.predictor.baselines import baseline_prediction
from av_semcom.models.predictor.data import (
    audit_predictor_samples,
    fit_audio_normalizer,
    select_predictor_samples,
)
from av_semcom.models.predictor.model import AudioToMotionGRU, masked_l1_loss
from av_semcom.models.reconstruction.backend import FakeReconstructionBackend


def _write_pair(
    root: Path,
    *,
    speaker: str,
    sample_id: str,
    split: str,
    audio_value: float,
) -> GridSample:
    frame_count = 5
    audio_path = root / f"grid/processed/audio/{speaker}/{sample_id}.npz"
    atomic_save_npz(
        audio_path,
        features=np.full((frame_count, 4, 80), audio_value, dtype=np.float32),
    )
    crops = np.zeros((frame_count, 256, 256, 3), dtype=np.uint8)
    for index in range(frame_count):
        crops[index].fill(min(index * 20 + int(audio_value), 255))
    sequence = FakeReconstructionBackend().extract_motion(
        crops,
        np.ones(frame_count, dtype=np.bool_),
        sample_id=sample_id,
        fps=25,
        config_fingerprint="test",
    )
    motion_path = root / f"grid/processed/motion/{speaker}/{sample_id}.npz"
    save_motion_sequence(motion_path, sequence)
    return GridSample(
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


def _grid_settings(root: Path) -> GridSettings:
    return GridSettings.from_config(
        {
            "root": str(root),
            "raw_video_dir": "grid/raw/video",
            "raw_audio_dir": "grid/raw/audio",
            "manifest_path": "grid/manifests/data.jsonl",
            "failure_dir": "grid/manifests/failures",
            "processed_dir": "grid/processed",
            "speakers": ["s1", "s2", "s3"],
            "max_samples": 10,
            "fps": 25,
            "split_seed": 42,
            "validation_ratio": 0.1,
            "test_ratio": 0.1,
            "resume": True,
        }
    )


def test_predictor_data_selection_audit_and_train_only_stats(tmp_path: Path) -> None:
    train = _write_pair(
        tmp_path,
        speaker="s3",
        sample_id="train",
        split="train",
        audio_value=1,
    )
    validation = _write_pair(
        tmp_path,
        speaker="s1",
        sample_id="validation",
        split="validation",
        audio_value=100,
    )
    test = _write_pair(
        tmp_path,
        speaker="s2",
        sample_id="test",
        split="test",
        audio_value=200,
    )
    incomplete = replace(train, sample_id="incomplete", motion_path=None)

    selected = select_predictor_samples(
        [incomplete, train, validation, test],
        _grid_settings(tmp_path),
    )
    audit = audit_predictor_samples(selected, tmp_path)
    normalizer = fit_audio_normalizer([train], tmp_path)

    assert [sample.sample_id for sample in selected] == ["train", "validation", "test"]
    assert audit.errors == ()
    assert audit.split_counts == {"train": 1, "validation": 1, "test": 1}
    assert np.allclose(normalizer.mean, 1)
    with pytest.raises(ValueError, match="train samples"):
        fit_audio_normalizer([train, validation], tmp_path)


def test_causal_gru_shape_and_future_independence() -> None:
    torch.manual_seed(42)
    model = AudioToMotionGRU(
        audio_projection_dim=16,
        hidden_dim=24,
        num_layers=2,
        dropout=0.1,
    )
    model.eval()
    first = torch.randn(2, 6, 4, 80)
    changed = first.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:])

    with torch.no_grad():
        first_output = model(first)
        changed_output = model(changed)

    assert first_output.shape == (2, 6, 18)
    assert torch.allclose(first_output[:, :3], changed_output[:, :3])


def test_masked_l1_and_baselines() -> None:
    target_tensor = torch.ones(1, 3, 18)
    prediction_tensor = torch.zeros_like(target_tensor)
    mask = torch.tensor([[True, False, True]])
    assert masked_l1_loss(prediction_tensor, target_tensor, mask).item() == pytest.approx(1)

    target = np.arange(54, dtype=np.float32).reshape(3, 18)
    mean = np.ones(18, dtype=np.float32)
    zero = baseline_prediction("zero_motion", target, mean)
    average = baseline_prediction("train_mean", target, mean)
    persistence = baseline_prediction("oracle_persistence", target, mean)
    assert np.all(zero == 0)
    assert np.all(average[0] == 0)
    assert np.all(average[1:] == 1)
    assert np.all(persistence[0] == 0)
    assert np.array_equal(persistence[1:], target[:-1])


def test_checkpoint_and_prediction_fingerprint_protection(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "best.pt"
    atomic_save_checkpoint(
        checkpoint_path,
        {"experiment_fingerprint": "expected", "model_state": {}},
    )
    assert (
        load_checkpoint(
            checkpoint_path,
            expected_fingerprint="expected",
        )["model_state"]
        == {}
    )
    with pytest.raises(ValueError, match="fingerprint"):
        load_checkpoint(checkpoint_path, expected_fingerprint="stale")

    prediction_path = tmp_path / "prediction.npz"
    values = np.zeros((5, 18), dtype=np.float32)
    save_prediction(
        prediction_path,
        sample_id="sample",
        method="zero_motion",
        split="test",
        speaker_id="s2",
        prediction=values,
        target=values,
        valid_mask=np.ones(5, dtype=np.bool_),
        seed=None,
        experiment_fingerprint="expected",
    )
    assert (
        load_prediction(
            prediction_path,
            expected_fingerprint="expected",
        )["sample_id"]
        == "sample"
    )
    with pytest.raises(ValueError, match="fingerprint"):
        load_prediction(prediction_path, expected_fingerprint="stale")


def test_motion_normalizer_fixture_uses_train_scope(tmp_path: Path) -> None:
    sample = _write_pair(
        tmp_path,
        speaker="s3",
        sample_id="train",
        split="train",
        audio_value=1,
    )
    from av_semcom.models.motion.sequence import load_motion_sequence

    sequence = load_motion_sequence(tmp_path / sample.motion_path)
    normalizer = fit_motion_normalizer([sequence], scope="train_stats")
    assert normalizer.scope == "train_stats"


def test_parallel_landmarks_use_one_backend_per_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = threading.Barrier(2)

    class ThreadBoundLandmarks:
        def __init__(self) -> None:
            self.owner = threading.get_ident()
            self.closed = False
            instances.append(self)

        def detect(self, rgb_image: np.ndarray) -> None:
            assert threading.get_ident() == self.owner
            barrier.wait(timeout=5)
            return None

        def close(self) -> None:
            self.closed = True

    instances: list[ThreadBoundLandmarks] = []
    monkeypatch.setattr(
        reconstruction,
        "MediaPipeFaceMeshBackend",
        ThreadBoundLandmarks,
    )
    landmarks = reconstruction._ThreadLocalMediaPipeFaceMeshBackend()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(landmarks.detect, (frame, frame)))
    landmarks.close()

    assert results == [None, None]
    assert len(instances) == 2
    assert all(instance.closed for instance in instances)
