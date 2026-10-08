"""`spike.py doctor`: check the environment without downloading any models."""
from __future__ import annotations

import importlib.metadata as md
import platform
import shutil
import sys
from pathlib import Path

from .util import ensure_ffmpeg, preload_torch, query_gpu_mb


def _line(ok: bool | None, label: str, detail: str = "") -> None:
    mark = {True: "OK  ", False: "FAIL", None: "info"}[ok]
    print(f"[{mark}] {label}{': ' + detail if detail else ''}")


def _ver(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def run_doctor() -> int:
    problems = 0
    _line(None, "python", f"{sys.version.split()[0]} on {platform.platform()}")
    if sys.version_info >= (3, 13):
        _line(None, "note", "Python 3.12 is the best-tested choice for this stack; 3.13 may hit missing wheels")

    torch = preload_torch()
    cuda_ok = False
    if torch is None:
        _line(False, "torch", "not installed. Install the CUDA build first (see README)")
        problems += 1
    else:
        cuda_ok = torch.cuda.is_available()
        _line(cuda_ok, "torch", f"{torch.__version__}, built for CUDA {torch.version.cuda}")
        if cuda_ok:
            free, total = torch.cuda.mem_get_info()
            _line(True, "GPU", f"{torch.cuda.get_device_name(0)}, {total / 2**30:.1f} GB total, {free / 2**30:.1f} GB free")
            try:
                a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
                float((a @ a).sum())
                _line(True, "GPU fp16 matmul smoke test")
            except Exception as exc:
                _line(False, "GPU fp16 matmul smoke test", str(exc))
                problems += 1
        else:
            _line(False, "CUDA", "torch cannot see a GPU. You probably installed the CPU-only torch wheel; "
                                 "reinstall from the CUDA index shown at pytorch.org")
            problems += 1
    smi = query_gpu_mb()
    _line(None if smi else False, "nvidia-smi", f"{smi[0]} MB used of {smi[1]} MB" if smi else "not found")

    try:
        import ctranslate2

        n = ctranslate2.get_cuda_device_count()
        types = sorted(ctranslate2.get_supported_compute_types("cuda")) if n else []
        _line(n > 0, "ctranslate2 (Whisper engine)", f"{ctranslate2.__version__}, {n} CUDA device(s), compute types {types}")
        if n == 0:
            _line(None, "hint", "if torch sees the GPU but this does not, the cuDNN/cuBLAS DLLs are missing: "
                                "install a CUDA 12.x build of torch (CTranslate2 expects CUDA 12)")
            problems += 1
    except Exception as exc:
        _line(False, "ctranslate2", str(exc))
        problems += 1

    for pkg in ("faster-whisper", "audio-separator", "transformers", "soundfile", "jiwer", "pykakasi", "numpy", "av",
                "imageio-ffmpeg"):
        v = _ver(pkg)
        _line(v is not None, pkg, v or "not installed (pip install -r requirements.txt)")
        problems += v is None
    try:
        from audio_separator.separator import Separator  # noqa: F401

        _line(True, "audio-separator import")
    except Exception as exc:
        _line(False, "audio-separator import", f"{type(exc).__name__}: {exc}")
        problems += 1
    try:
        import pykakasi

        got = pykakasi.kakasi().convert("夜")[0]["hepburn"]
        _line(got == "yoru", "Japanese romanization", f"夜 -> {got}")
    except Exception as exc:
        _line(False, "Japanese romanization", str(exc))
        problems += 1

    try:
        _line(True, "ffmpeg", ensure_ffmpeg())
    except Exception as exc:
        _line(False, "ffmpeg", str(exc))
        problems += 1

    for label, path in (("disk free (here)", Path.cwd()), ("disk free (home, model cache)", Path.home())):
        free = shutil.disk_usage(path).free / 2**30
        _line(free > 12, label, f"{free:.0f} GB (first run downloads about 5 GB of models)")
    print()
    print("All good." if problems == 0 else f"{problems} problem(s) found; fix them before `spike.py run`.")
    return 1 if problems else 0
