"""Small helpers: console setup, logging, stage timing, GPU monitoring, CUDA DLL preloading."""
from __future__ import annotations

import gc
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager


def setup_console() -> None:
    """Make printing Japanese safe on Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_dur(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}m{s:02d}s"


def preload_torch():
    """Import torch before CTranslate2 so its bundled CUDA/cuDNN DLLs are already loaded (Windows)."""
    try:
        import torch  # noqa: F401

        return torch
    except Exception:
        return None


def free_gpu() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def cuda_available() -> bool:
    torch = preload_torch()
    try:
        return bool(torch and torch.cuda.is_available())
    except Exception:
        return False


def query_gpu_mb():
    """Return (used_mb, total_mb) from nvidia-smi, or None when unavailable."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if res.returncode != 0:
            return None
        return parse_smi(res.stdout)
    except Exception:
        return None


def parse_smi(text: str):
    first = text.strip().splitlines()[0]
    used, total = (int(x.strip()) for x in first.split(",")[:2])
    return used, total


class GpuMonitor:
    """Polls nvidia-smi in a thread to record peak GPU memory use (all processes, approximate)."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        first = query_gpu_mb()
        self.available = first is not None
        self.total_mb = first[1] if first else None
        self.baseline_mb = first[0] if first else None
        self.peak_mb = first[0] if first else 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        if self.available:
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def reset_peak(self) -> None:
        q = query_gpu_mb() if self.available else None
        self.peak_mb = q[0] if q else 0

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            q = query_gpu_mb()
            if q:
                self.peak_mb = max(self.peak_mb, q[0])


class Stages:
    """Records wall time and peak GPU memory for each pipeline stage."""

    def __init__(self, monitor: GpuMonitor):
        self.monitor = monitor
        self.rows: list[tuple[str, float, int | None]] = []

    @contextmanager
    def stage(self, name: str):
        log(f">> {name}")
        self.monitor.reset_peak()
        start = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - start
            peak = None
            if self.monitor.available:
                q = query_gpu_mb()
                peak = max(self.monitor.peak_mb, q[0] if q else 0)
            self.rows.append((name, dt, peak))
            extra = f", peak GPU memory in use {peak} MB of {self.monitor.total_mb} MB" if peak is not None else ""
            log(f"ok {name}: {fmt_dur(dt)}{extra}")


def ensure_ffmpeg(cache_dir=None) -> str:
    """Make an `ffmpeg` command available on PATH (audio-separator refuses to start without one).

    Uses the system ffmpeg when present, otherwise the static binary bundled with `imageio-ffmpeg`,
    copied under the name `ffmpeg` into a cache folder that is prepended to PATH.
    Returns the resolved path, or raises RuntimeError with install instructions.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("ffmpeg not found. Install it (Windows: `winget install Gyan.FFmpeg`, then reopen the "
                           "terminal) or `pip install imageio-ffmpeg`.") from exc
    from pathlib import Path

    target_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "lyricspike" / "bin"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not target.exists():
        shutil.copyfile(exe, target)
        target.chmod(0o755)
    os.environ["PATH"] = str(target_dir) + os.pathsep + os.environ.get("PATH", "")
    return str(target)
