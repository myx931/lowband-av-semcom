"""Deterministic speaker-isolated dataset splitting."""

from __future__ import annotations

import random
from collections.abc import Iterable


def assign_speaker_splits(
    speaker_ids: Iterable[str],
    *,
    seed: int,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, str]:
    """Assign speakers to train, validation, and test without identity overlap.

    A single speaker is explicitly treated as a pipeline-only ``pilot`` split.
    Formal splitting requires at least three speakers so every split is non-empty.
    """

    speakers = sorted(set(speaker_ids))
    if not speakers:
        raise ValueError("at least one speaker is required")
    if len(speakers) == 1:
        return {speakers[0]: "pilot"}
    if len(speakers) < 3:
        raise ValueError("formal speaker-isolated splitting requires at least three speakers")
    if not 0 < validation_ratio < 1 or not 0 < test_ratio < 1:
        raise ValueError("validation_ratio and test_ratio must be between 0 and 1")
    if validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio + test_ratio must be less than 1")

    random.Random(seed).shuffle(speakers)
    validation_count = max(1, round(len(speakers) * validation_ratio))
    test_count = max(1, round(len(speakers) * test_ratio))
    if validation_count + test_count >= len(speakers):
        validation_count = 1
        test_count = 1

    mapping: dict[str, str] = {}
    for speaker in speakers[:test_count]:
        mapping[speaker] = "test"
    for speaker in speakers[test_count : test_count + validation_count]:
        mapping[speaker] = "validation"
    for speaker in speakers[test_count + validation_count :]:
        mapping[speaker] = "train"
    return mapping
