"""Read-only progress inspection for resumable reconstruction runs."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReconstructionProgress:
    """Snapshot of one audio-to-motion reconstruction evaluation."""

    run_dir: str
    completed_samples: int
    total_samples: int
    percent_complete: float
    failed_samples: int
    status: str
    samples_per_minute: float | None
    eta_seconds: float | None
    summary_available: bool
    gpu_status: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot."""

        return asdict(self)


def inspect_reconstruction_progress(run_dir: Path) -> ReconstructionProgress:
    """Inspect artifacts without changing or locking the active run."""

    reconstruction_dir = run_dir / "reconstruction"
    runtime = _read_json(reconstruction_dir / "runtime.json")
    total = int(runtime.get("evaluation_sample_count", 0))
    sample_paths = sorted(
        (reconstruction_dir / "samples").glob("*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    completed = len(sample_paths)
    failed = _count_nonempty_lines(reconstruction_dir / "failures.jsonl")
    complete = (reconstruction_dir / "complete.json").is_file()
    summary_available = (reconstruction_dir / "summary.json").is_file()
    rate = _recent_completion_rate(sample_paths)
    remaining = max(total - completed, 0)
    eta = remaining / (rate / 60.0) if rate and remaining else (0.0 if complete else None)
    if complete:
        status = "complete"
    elif _reconstruction_process_running(run_dir):
        status = "running"
    elif completed or runtime:
        status = "interrupted"
    else:
        status = "not_started"
    percent = 100.0 * completed / total if total else 0.0
    return ReconstructionProgress(
        run_dir=str(run_dir),
        completed_samples=completed,
        total_samples=total,
        percent_complete=percent,
        failed_samples=failed,
        status=status,
        samples_per_minute=rate,
        eta_seconds=eta,
        summary_available=summary_available,
        gpu_status=_gpu_status(),
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _count_nonempty_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def _recent_completion_rate(paths: list[Path], window: int = 20) -> float | None:
    recent = paths[-window:]
    if len(recent) < 2:
        return None
    elapsed = recent[-1].stat().st_mtime - recent[0].stat().st_mtime
    if elapsed <= 0:
        return None
    return 60.0 * (len(recent) - 1) / elapsed


def _reconstruction_process_running(run_dir: Path) -> bool:
    proc = Path("/proc")
    if not proc.is_dir():
        return False
    target = str(run_dir)
    for command_path in proc.glob("[0-9]*/cmdline"):
        try:
            command = command_path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, PermissionError):
            continue
        if "reconstruct_audio_to_motion.py" in command and target in command:
            return True
    return False


def _gpu_status() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def format_duration(seconds: float | None) -> str:
    """Format an ETA for terminal output."""

    if seconds is None:
        return "unknown"
    rounded = max(int(seconds), 0)
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def wait_for_next_snapshot(interval_seconds: float) -> None:
    """Sleep between explicit watch-mode snapshots."""

    time.sleep(interval_seconds)
