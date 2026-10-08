"""Audio decoding (any format PyAV/ffmpeg can read) and vocal-activity segmentation."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def decode(path: str | Path, sr: int, stereo: bool = False) -> np.ndarray:
    """Decode any audio file (mp3, flac, wav, m4a, ogg, ...) to float32 at `sr` Hz.

    Returns [n] for mono or [n, 2] for stereo. Uses PyAV directly (bundled ffmpeg, no system install needed).
    Deliberately avoids faster_whisper.audio.decode_audio, which breaks with the newest PyAV releases.
    """
    import av

    channels = 2 if stereo else 1
    resampler = av.audio.resampler.AudioResampler(format="flt", layout="stereo" if stereo else "mono", rate=sr)
    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"no audio stream found in {path}")
        for frame in container.decode(container.streams.audio[0]):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):  # flush
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError(f"could not decode any audio from {path}")
    data = np.concatenate(chunks).astype(np.float32)
    data = data[: len(data) // channels * channels]
    return data.reshape(-1, 2) if stereo else data


def write_wav(path: str | Path, data: np.ndarray, sr: int) -> None:
    import soundfile as sf

    sf.write(str(path), data, sr, subtype="PCM_16")


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data, sr


def _runs(mask: np.ndarray) -> list[list[int]]:
    """[start, end) index pairs of consecutive True values."""
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [[int(s), int(e)] for s, e in zip(edges[0::2], edges[1::2])]


def _frame_db(wave: np.ndarray, sr: int, frame_s: float) -> np.ndarray:
    """RMS level in dB for consecutive frames of `frame_s` seconds."""
    frame = int(frame_s * sr)
    n_frames = len(wave) // frame
    if n_frames == 0:
        return np.zeros(0)
    frames = wave[: n_frames * frame].reshape(n_frames, frame).astype(np.float64)
    return 20 * np.log10(np.sqrt((frames**2).mean(axis=1) + 1e-12))


def describe_levels(wave: np.ndarray, sr: int = 16000, frame_s: float = 0.05, rel_db: float = 30.0,
                    floor_db: float = -60.0) -> dict:
    """How loud the stem is and where `vocal_segments` puts its 'singing starts here' threshold."""
    db = _frame_db(wave, sr, frame_s)
    if len(db) == 0:
        return {"p95_db": None, "threshold_db": None, "active_fraction": 0.0}
    p95 = float(np.percentile(db, 95))
    threshold = max(p95 - rel_db, floor_db)
    return {"p95_db": p95, "threshold_db": threshold, "active_fraction": float((db > threshold).mean())}


def vocal_segments(
    wave: np.ndarray,
    sr: int = 16000,
    frame_s: float = 0.05,
    rel_db: float = 30.0,
    floor_db: float = -60.0,
    min_gap_s: float = 0.8,
    min_dur_s: float = 0.3,
    pad_s: float = 0.2,
    max_len_s: float = 25.0,
) -> list[tuple[float, float]]:
    """Find sung regions in an (already separated) vocal stem using frame energy.

    Energy is compared with the 95th-percentile level of the stem, so the threshold adapts to loudness.
    Short pauses are bridged, tiny blips dropped, and long regions split at their quietest point so
    each piece fits comfortably in Whisper's 30 s window.
    """
    n = len(wave)
    duration = n / sr
    db = _frame_db(wave, sr, frame_s)
    if len(db) == 0:
        return []
    threshold = max(float(np.percentile(db, 95)) - rel_db, floor_db)
    runs = _runs(db > threshold)

    gap = int(round(min_gap_s / frame_s))
    merged: list[list[int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    min_frames = int(round(min_dur_s / frame_s))
    max_frames = int(max_len_s / frame_s)
    pieces: list[tuple[int, int]] = []
    for s, e in merged:
        if e - s < min_frames:
            continue
        while e - s > max_frames:
            lo, hi = s + int(max_frames * 0.6), s + max_frames
            cut = lo + int(np.argmin(db[lo:hi]))
            pieces.append((s, cut))
            s = cut
        pieces.append((s, e))

    segs = [[max(0.0, s * frame_s - pad_s), min(duration, e * frame_s + pad_s)] for s, e in pieces]
    for i in range(1, len(segs)):  # padding must not make neighbours overlap
        if segs[i][0] < segs[i - 1][1]:
            mid = (segs[i][0] + segs[i - 1][1]) / 2
            segs[i][0] = segs[i - 1][1] = mid
    return [(a, b) for a, b in segs if b - a > 0.05]
