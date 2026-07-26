"""Injectable motion extraction and portrait reconstruction backends."""

from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from av_semcom.models.motion.sequence import (
    LIVEPORTRAIT_LIP_INDICES,
    MotionSequence,
)

ReconstructionMode = Literal["full_motion", "lip_only"]


class ReconstructionBackend(Protocol):
    """Minimal interface used by extraction and sensitivity experiments."""

    @property
    def name(self) -> str:
        """Stable backend identifier."""

    @property
    def revision(self) -> str:
        """Pinned source revision."""

    def extract_motion(
        self,
        crops: np.ndarray,
        valid_mask: np.ndarray,
        *,
        sample_id: str,
        fps: float,
        config_fingerprint: str,
    ) -> MotionSequence:
        """Extract a typed motion sequence from RGB face crops."""

    def reconstruct(
        self,
        source_frame: np.ndarray,
        sequence: MotionSequence,
        *,
        mode: ReconstructionMode,
        lip_vector: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render RGB frames from a reference face and motion."""

    def close(self) -> None:
        """Release backend resources."""


class FakeReconstructionBackend:
    """Small deterministic backend for CPU tests and smoke pipelines."""

    name = "fake"
    revision = "test-v1"

    def extract_motion(
        self,
        crops: np.ndarray,
        valid_mask: np.ndarray,
        *,
        sample_id: str,
        fps: float,
        config_fingerprint: str,
    ) -> MotionSequence:
        """Derive deterministic pseudo-expression from frame brightness."""

        _validate_crops(crops, valid_mask)
        source_frame_index = int(np.flatnonzero(valid_mask)[0])
        frame_count = crops.shape[0]
        expression = np.zeros((frame_count, 21, 3), dtype=np.float32)
        brightness = crops.mean(axis=(1, 2, 3), dtype=np.float64).astype(np.float32) / 255.0
        centered = brightness - brightness[source_frame_index]
        lip_indices = np.asarray(LIVEPORTRAIT_LIP_INDICES, dtype=np.int64)
        expression[:, lip_indices, 1] = centered[:, None]
        expression[:, lip_indices, 2] = centered[:, None] * 0.5
        lip_delta = expression[:, lip_indices] - expression[source_frame_index, lip_indices]
        rotation = np.repeat(np.eye(3, dtype=np.float32)[None], frame_count, axis=0)
        translation = np.zeros((frame_count, 3), dtype=np.float32)
        scale = np.ones((frame_count, 1), dtype=np.float32)
        canonical = np.zeros((frame_count, 21, 3), dtype=np.float32)
        canonical[:, :, 0] = np.linspace(-0.5, 0.5, 21, dtype=np.float32)
        return MotionSequence(
            sample_id=sample_id,
            fps=fps,
            backend=self.name,
            backend_revision=self.revision,
            config_fingerprint=config_fingerprint,
            source_frame_index=source_frame_index,
            expression=expression,
            lip_delta=lip_delta.astype(np.float32),
            rotation=rotation,
            translation=translation,
            scale=scale,
            canonical_keypoints=canonical,
            valid_mask=valid_mask.astype(np.bool_),
            lip_indices=lip_indices,
        )

    def reconstruct(
        self,
        source_frame: np.ndarray,
        sequence: MotionSequence,
        *,
        mode: ReconstructionMode,
        lip_vector: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render a colored lower-face patch controlled by pseudo-motion."""

        _validate_source_frame(source_frame)
        active = sequence.lip_vector if lip_vector is None else lip_vector
        if active.shape != (sequence.frame_count, 18):
            raise ValueError("lip_vector shape does not match the motion sequence")
        if mode not in {"full_motion", "lip_only"}:
            raise ValueError(f"unsupported reconstruction mode: {mode}")
        frames = np.repeat(source_frame[None], sequence.frame_count, axis=0)
        height, width = source_frame.shape[:2]
        y0, y1 = height * 5 // 8, height * 7 // 8
        x0, x1 = width * 3 // 8, width * 5 // 8
        strength = np.clip(np.abs(active).mean(axis=1) * 2550, 0, 80).astype(np.uint8)
        for index, value in enumerate(strength):
            patch = frames[index, y0:y1, x0:x1].astype(np.int16)
            patch[..., 0] = np.clip(patch[..., 0] + int(value), 0, 255)
            frames[index, y0:y1, x0:x1] = patch.astype(np.uint8)
        return frames

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class LivePortraitBackendConfig:
    """Local paths and runtime controls for the pinned third-party backend."""

    repository: Path
    model_root: Path
    expected_revision: str
    device: str = "cuda:0"
    half_precision: bool = True
    stitching: bool = True
    reconstruction_batch_size: int = 16


class LivePortraitBackend:
    """Frozen LivePortrait motion extractor and renderer.

    This backend intentionally imports the submodule lazily so the core package
    and all CPU tests remain independent of its CUDA-specific environment.
    """

    name = "liveportrait"

    def __init__(self, config: LivePortraitBackendConfig) -> None:
        self._config = config
        self._revision = _verify_liveportrait_installation(config)
        repository_text = str(config.repository.resolve())
        if repository_text not in sys.path:
            sys.path.insert(0, repository_text)
        try:
            inference_module = importlib.import_module("src.config.inference_config")
            wrapper_module = importlib.import_module("src.live_portrait_wrapper")
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Could not import the pinned LivePortrait submodule. Run this command "
                "inside the dedicated Python 3.10 LivePortrait environment."
            ) from exc

        weights = config.model_root.resolve()
        inference_config = inference_module.InferenceConfig(
            models_config=str(config.repository / "src/config/models.yaml"),
            checkpoint_F=str(weights / "liveportrait/base_models/appearance_feature_extractor.pth"),
            checkpoint_M=str(weights / "liveportrait/base_models/motion_extractor.pth"),
            checkpoint_G=str(weights / "liveportrait/base_models/spade_generator.pth"),
            checkpoint_W=str(weights / "liveportrait/base_models/warping_module.pth"),
            checkpoint_S=str(
                weights / "liveportrait/retargeting_models/stitching_retargeting_module.pth"
            ),
            flag_force_cpu=config.device == "cpu",
            device_id=_device_id(config.device),
            flag_use_half_precision=config.half_precision and config.device != "cpu",
            flag_stitching=config.stitching,
            flag_do_crop=False,
            flag_pasteback=False,
        )
        try:
            self._wrapper = wrapper_module.LivePortraitWrapper(inference_config)
            camera_module = importlib.import_module("src.utils.camera")
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"Could not initialize LivePortrait: {exc}") from exc
        self._rotation_matrix = camera_module.get_rotation_matrix

    @property
    def revision(self) -> str:
        return self._revision

    def extract_motion(
        self,
        crops: np.ndarray,
        valid_mask: np.ndarray,
        *,
        sample_id: str,
        fps: float,
        config_fingerprint: str,
    ) -> MotionSequence:
        """Extract full and mouth-only motion from pre-cropped RGB frames."""

        _validate_crops(crops, valid_mask)
        source_frame_index = int(np.flatnonzero(valid_mask)[0])
        expression: list[np.ndarray] = []
        rotation: list[np.ndarray] = []
        translation: list[np.ndarray] = []
        scale: list[np.ndarray] = []
        canonical: list[np.ndarray] = []
        for crop in crops:
            prepared = self._wrapper.prepare_source(crop)
            info = self._wrapper.get_kp_info(prepared)
            expression.append(info["exp"].detach().cpu().numpy()[0])
            translation.append(info["t"].detach().cpu().numpy()[0])
            scale.append(info["scale"].detach().cpu().numpy().reshape(-1)[:1])
            canonical.append(info["kp"].detach().cpu().numpy()[0])
            matrix = self._rotation_matrix(info["pitch"], info["yaw"], info["roll"])
            rotation.append(matrix.detach().cpu().numpy()[0])

        expression_array = np.asarray(expression, dtype=np.float32)
        lip_indices = np.asarray(LIVEPORTRAIT_LIP_INDICES, dtype=np.int64)
        lip_delta = (
            expression_array[:, lip_indices]
            - expression_array[source_frame_index, lip_indices][None]
        )
        return MotionSequence(
            sample_id=sample_id,
            fps=fps,
            backend=self.name,
            backend_revision=self.revision,
            config_fingerprint=config_fingerprint,
            source_frame_index=source_frame_index,
            expression=expression_array,
            lip_delta=lip_delta.astype(np.float32),
            rotation=np.asarray(rotation, dtype=np.float32),
            translation=np.asarray(translation, dtype=np.float32),
            scale=np.asarray(scale, dtype=np.float32),
            canonical_keypoints=np.asarray(canonical, dtype=np.float32),
            valid_mask=valid_mask.astype(np.bool_),
            lip_indices=lip_indices,
        )

    def reconstruct(
        self,
        source_frame: np.ndarray,
        sequence: MotionSequence,
        *,
        mode: ReconstructionMode,
        lip_vector: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render full target motion or source-relative mouth-only motion."""

        _validate_source_frame(source_frame)
        if sequence.backend != self.name:
            raise ValueError(
                f"motion backend {sequence.backend!r} is incompatible with {self.name!r}"
            )
        if sequence.backend_revision != self.revision:
            raise ValueError("motion artifact was extracted with another backend revision")

        import torch

        source = self._wrapper.prepare_source(source_frame)
        source_info = self._wrapper.get_kp_info(source)
        source_keypoints = self._wrapper.transform_keypoint(source_info)
        source_feature = self._wrapper.extract_feature_3d(source)
        active_lip = sequence.lip_vector if lip_vector is None else lip_vector
        if active_lip.shape != (sequence.frame_count, 18):
            raise ValueError("lip_vector shape does not match the motion sequence")

        rendered: list[np.ndarray] = []
        batch_size = self._config.reconstruction_batch_size
        for batch_start in range(0, sequence.frame_count, batch_size):
            batch_end = min(batch_start + batch_size, sequence.frame_count)
            active_batch_size = batch_end - batch_start
            source_keypoints_batch = source_keypoints.repeat(active_batch_size, 1, 1)
            source_feature_batch = source_feature.repeat(
                active_batch_size,
                *([1] * (source_feature.ndim - 1)),
            )
            if mode == "full_motion":
                canonical = torch.from_numpy(
                    sequence.canonical_keypoints[batch_start:batch_end]
                ).to(self._wrapper.device)
                rotation = torch.from_numpy(sequence.rotation[batch_start:batch_end]).to(
                    self._wrapper.device
                )
                expression = torch.from_numpy(sequence.expression[batch_start:batch_end]).to(
                    self._wrapper.device
                )
                translation = torch.from_numpy(sequence.translation[batch_start:batch_end]).to(
                    self._wrapper.device
                )
                scale = torch.from_numpy(sequence.scale[batch_start:batch_end]).to(
                    self._wrapper.device
                )
            elif mode == "lip_only":
                canonical = source_info["kp"].repeat(active_batch_size, 1, 1)
                rotation = self._rotation_matrix(
                    source_info["pitch"],
                    source_info["yaw"],
                    source_info["roll"],
                ).repeat(active_batch_size, 1, 1)
                expression = source_info["exp"].repeat(active_batch_size, 1, 1)
                lip_delta = torch.from_numpy(
                    active_lip[batch_start:batch_end].reshape(active_batch_size, 6, 3)
                ).to(self._wrapper.device)
                expression[:, sequence.lip_indices.tolist(), :] += lip_delta
                translation = source_info["t"].repeat(active_batch_size, 1)
                scale = source_info["scale"].repeat(active_batch_size, 1)
            else:
                raise ValueError(f"unsupported reconstruction mode: {mode}")

            driving = canonical @ rotation + expression
            driving *= scale[..., None]
            driving[:, :, :2] += translation[:, None, :2]
            if self._config.stitching:
                driving = self._wrapper.stitching(source_keypoints_batch, driving)
            result = self._wrapper.warp_decode(
                source_feature_batch,
                source_keypoints_batch,
                driving,
            )
            for output in self._wrapper.parse_output(result["out"]):
                rendered.append(_resize_output(output, source_frame.shape))
        return np.asarray(rendered, dtype=np.uint8)

    def close(self) -> None:
        self._wrapper = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _validate_crops(crops: np.ndarray, valid_mask: np.ndarray) -> None:
    if crops.ndim != 4 or crops.shape[1:] != (256, 256, 3):
        raise ValueError(f"crops must have shape [T, 256, 256, 3], got {crops.shape}")
    if crops.dtype != np.uint8:
        raise ValueError("crops must use uint8 RGB values")
    if valid_mask.shape != (crops.shape[0],):
        raise ValueError("valid_mask must have shape [T]")
    if not valid_mask.any():
        raise ValueError("at least one valid source frame is required")


def _validate_source_frame(source_frame: np.ndarray) -> None:
    if source_frame.shape != (256, 256, 3) or source_frame.dtype != np.uint8:
        raise ValueError("source_frame must be a uint8 RGB image with shape [256, 256, 3]")


def _resize_output(output: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    if output.shape == target_shape:
        return output.astype(np.uint8, copy=False)
    from PIL import Image

    return np.asarray(
        Image.fromarray(output).resize(
            (target_shape[1], target_shape[0]),
            resample=Image.Resampling.LANCZOS,
        ),
        dtype=np.uint8,
    )


def _device_id(device: str) -> int:
    if device == "cpu":
        return 0
    if not device.startswith("cuda:"):
        raise ValueError("LivePortrait device must be 'cpu' or 'cuda:<index>'")
    try:
        return int(device.split(":", maxsplit=1)[1])
    except ValueError as exc:
        raise ValueError("LivePortrait CUDA device index must be an integer") from exc


def _verify_liveportrait_installation(config: LivePortraitBackendConfig) -> str:
    if not (config.repository / "src/live_portrait_wrapper.py").is_file():
        raise RuntimeError(
            "LivePortrait submodule is missing. Run `git submodule update --init --recursive`."
        )
    try:
        revision = subprocess.run(
            ["git", "-C", str(config.repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Could not inspect LivePortrait revision: {exc}") from exc
    if revision != config.expected_revision:
        raise RuntimeError(
            f"LivePortrait revision mismatch: expected {config.expected_revision}, found {revision}"
        )
    required = (
        "liveportrait/base_models/appearance_feature_extractor.pth",
        "liveportrait/base_models/motion_extractor.pth",
        "liveportrait/base_models/spade_generator.pth",
        "liveportrait/base_models/warping_module.pth",
        "liveportrait/retargeting_models/stitching_retargeting_module.pth",
    )
    missing = [relative for relative in required if not (config.model_root / relative).is_file()]
    if missing:
        raise RuntimeError(
            "LivePortrait weights are missing under MODEL_ROOT: " + ", ".join(missing)
        )
    return revision
