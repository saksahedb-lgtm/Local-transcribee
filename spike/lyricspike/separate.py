"""Vocal separation via the audio-separator package (Roformer checkpoints), run one model at a time."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np

from . import audio
from .util import ensure_ffmpeg, free_gpu, log

# Reasonable starting points; any filename that audio-separator supports works with --sep-model.
DEFAULT_VOCAL_MODEL = "vocals_mel_band_roformer.ckpt"
SECOND_VOCAL_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
DEFAULT_KARAOKE_MODEL = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"

_STEM_RE = re.compile(r"_\(([^)]+)\)")


def run_model(input_wav: Path, out_dir: Path, model_filename: str, model_dir: Path,
              precision: str = "fp16", segment_size: int | None = None,
              overlap: int | None = None) -> dict[str, Path]:
    """Run one separation model; returns {lowercase stem name: wav path}. Frees GPU memory afterwards."""
    ensure_ffmpeg()
    from audio_separator.separator import Separator

    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict = dict(output_dir=str(out_dir), model_file_dir=str(model_dir), output_format="WAV")
    if precision == "fp16":
        kwargs["use_native_fp16"] = True
    elif precision == "autocast":
        kwargs["use_autocast"] = True
    if segment_size or overlap:
        kwargs["mdxc_params"] = {
            "segment_size": segment_size or 256,
            "override_model_segment_size": bool(segment_size),
            "batch_size": 1,
            "overlap": overlap,
            "pitch_shift": 0,
        }
    sep = Separator(**kwargs)
    try:
        sep.load_model(model_filename=model_filename)
        files = sep.separate(str(input_wav))
    finally:
        del sep
        free_gpu()

    stems: dict[str, Path] = {}
    for f in files:
        p = Path(f)
        p = p if p.is_absolute() else out_dir / p
        m = _STEM_RE.search(p.name)
        stems[(m.group(1) if m else p.stem).lower()] = p
    return stems


def _cached(out_dir: Path) -> dict[str, Path]:
    stems: dict[str, Path] = {}
    if out_dir.is_dir():
        for p in out_dir.glob("*.wav"):
            m = _STEM_RE.search(p.name)
            if m:
                stems[m.group(1).lower()] = p
    return stems


def _short(model_filename: str) -> str:
    return Path(model_filename).stem


def separate_vocals(input_wav: Path, stems_dir: Path, models: list[str], model_dir: Path,
                    precision: str = "fp16", force: bool = False, segment_size: int | None = None,
                    overlap: int | None = None) -> Path:
    """Produce stems_dir/vocals.wav. With several models their vocal stems are averaged (simple ensemble)."""
    target = stems_dir / "vocals.wav"
    key = "_".join(_short(m) for m in models)
    marker = stems_dir / "vocals.models.txt"
    if target.exists() and not force and marker.exists() and marker.read_text().strip() == key:
        log(f"using cached {target.name} ({key})")
        return target

    vocal_paths: list[Path] = []
    for model in models:
        model_out = stems_dir / _short(model)
        stems = _cached(model_out) if not force else {}
        if "vocals" not in stems:
            log(f"separating with {model} (first run downloads the model)")
            stems = run_model(input_wav, model_out, model, model_dir, precision, segment_size, overlap)
        if "vocals" not in stems:
            raise RuntimeError(f"{model} produced stems {sorted(stems)}; expected one named 'vocals'. "
                               "Pick a vocal model (see README).")
        vocal_paths.append(stems["vocals"])

    if len(vocal_paths) == 1:
        shutil.copyfile(vocal_paths[0], target)
    else:
        log(f"averaging {len(vocal_paths)} vocal stems (ensemble)")
        arrays = [audio.read_wav(p)[0] for p in vocal_paths]
        n = min(a.shape[0] for a in arrays)
        mix = np.mean([a[:n] for a in arrays], axis=0)
        peak = float(np.abs(mix).max())
        if peak > 0:
            mix = mix * (0.9 / peak)
        audio.write_wav(target, mix, 44100)
    marker.write_text(key)
    return target


# Karaoke-style models are trained with the *karaoke track* (music + backing vocals, lead removed) as their
# target, so the stem called "karaoke"/"instrumental" is the BACKING side and the leftover ("other") is the LEAD.
_BACKING_NAMES = {"karaoke", "instrumental", "inst", "no vocals", "no_vocals", "novocals"}
_LEAD_NAMES = {"vocals", "lead", "other", "no karaoke", "no_karaoke"}


def _rms_db(path: Path) -> float:
    data, _sr = audio.read_wav(path)
    return float(20 * np.log10(np.sqrt(np.mean(data.astype(np.float64) ** 2)) + 1e-9))


def classify_karaoke_stems(stems: dict[str, Path]) -> tuple[str, str, str]:
    """Return (lead_key, backing_key, how) for a two-stem karaoke model output."""
    if len(stems) != 2:
        raise RuntimeError(f"Expected 2 stems from the karaoke model, got {sorted(stems)}.")
    a, b = list(stems)
    if a in _BACKING_NAMES and b not in _BACKING_NAMES:
        return b, a, "by stem name"
    if b in _BACKING_NAMES and a not in _BACKING_NAMES:
        return a, b, "by stem name"
    if a in _LEAD_NAMES and b not in _LEAD_NAMES:
        return a, b, "by stem name"
    if b in _LEAD_NAMES and a not in _LEAD_NAMES:
        return b, a, "by stem name"
    loud_a, loud_b = _rms_db(stems[a]), _rms_db(stems[b])
    return (a, b, "by loudness (names were ambiguous)") if loud_a >= loud_b else (b, a, "by loudness (names were ambiguous)")


def split_lead_backing(vocals_wav: Path, stems_dir: Path, model: str, model_dir: Path,
                       precision: str = "fp16", force: bool = False) -> tuple[Path, Path]:
    """Split a vocal stem into lead and backing vocals with a karaoke-style model."""
    lead, backing = stems_dir / "lead.wav", stems_dir / "backing.wav"
    if lead.exists() and backing.exists() and not force:
        log("using cached lead/backing stems")
        return lead, backing
    model_out = stems_dir / ("karaoke_" + _short(model))
    stems = _cached(model_out) if not force else {}
    if len(stems) < 2:
        stems = run_model(vocals_wav, model_out, model, model_dir, precision)
    lead_key, backing_key, how = classify_karaoke_stems(stems)
    lead_db, backing_db = _rms_db(stems[lead_key]), _rms_db(stems[backing_key])
    log(f"karaoke split: lead = '{lead_key}' ({lead_db:.1f} dB), backing = '{backing_key}' ({backing_db:.1f} dB), "
        f"decided {how}. VERIFY BY EAR in viewer.html - if they are swapped, tell me.")
    if lead_db < backing_db - 6:
        log("WARNING: the stem chosen as lead is much quieter than the backing; the assignment may be swapped.")
    shutil.copyfile(stems[lead_key], lead)
    shutil.copyfile(stems[backing_key], backing)
    return lead, backing
