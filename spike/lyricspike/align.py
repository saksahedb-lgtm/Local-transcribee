"""Forced alignment of lyrics to singing.

Text is romanized to a-z/' (English as-is, Japanese via pykakasi), a CTC model (an MMS wav2vec2 aligner)
produces per-frame character probabilities, and a Viterbi pass finds the best monotonic placement of every
character. Word/line times fall out of that. The numerical parts are plain numpy so they are unit-testable
without a GPU; only `load_aligner` touches torch/transformers.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

SR = 16000
STRIDE_S = 0.02  # wav2vec2: one frame per 320 samples at 16 kHz
DEFAULT_ALIGNER = "MahmoudAshraf/mms-300m-1130-forced-aligner"

_JP = re.compile(r"[぀-ヿ㐀-䶿一-鿿ｦ-ﾟ]")
_DIGIT_WORDS = "zero one two three four five six seven eight nine".split()
_NEG = -1e30


class AlignmentError(RuntimeError):
    pass


# --------------------------------------------------------------------------- text -> units

@dataclass
class Unit:
    """A display unit (an English word or a Japanese segment) and the romanized text used to align it."""
    text: str
    roman: str
    start: float | None = None
    end: float | None = None
    score: float | None = None


@dataclass
class Line:
    units: list[Unit]
    text: str

    @property
    def start(self) -> float | None:
        return next((u.start for u in self.units if u.start is not None), None)

    @property
    def end(self) -> float | None:
        return next((u.end for u in reversed(self.units) if u.end is not None), None)

    @property
    def score(self) -> float | None:
        vals = [u.score for u in self.units if u.score is not None]
        return float(np.mean(vals)) if vals else None


_kakasi = None


def _get_kakasi():
    global _kakasi
    if _kakasi is None:
        import pykakasi

        _kakasi = pykakasi.kakasi()
    return _kakasi


def romanize_latin(text: str) -> str:
    """Lowercase a-z and apostrophes only; accents stripped, digits spelled digit by digit."""
    text = unicodedata.normalize("NFKD", text.lower())
    out = []
    for ch in text:
        if unicodedata.combining(ch):
            continue
        if "a" <= ch <= "z" or ch == "'":
            out.append(ch)
        elif ch in "’‘`´":
            out.append("'")
        elif ch.isdigit():
            try:
                out.append(_DIGIT_WORDS[unicodedata.digit(ch)])
            except (ValueError, IndexError):
                pass
    return "".join(out)


def has_japanese(text: str) -> bool:
    return bool(_JP.search(text))


def romanize_text(text: str) -> str:
    """Whole-string romanization, spaces removed (used for the phonetic error metric)."""
    parts = []
    for chunk in text.split():
        if has_japanese(chunk):
            parts.extend(romanize_latin(item["hepburn"]) for item in _get_kakasi().convert(chunk))
        else:
            parts.append(romanize_latin(chunk))
    return "".join(parts)


def tokenize_line(line: str) -> list[Unit]:
    """Split a lyric line into alignable units. Symbols with no sound are attached to a neighbour."""
    raw: list[tuple[Unit, str]] = []  # (unit, whitespace that preceded it in the source)
    for ci, chunk in enumerate(line.split()):
        sep = " " if ci else ""
        if has_japanese(chunk):
            for ji, item in enumerate(_get_kakasi().convert(chunk)):
                raw.append((Unit(item["orig"], romanize_latin(item["hepburn"])), sep if ji == 0 else ""))
        else:
            raw.append((Unit(chunk, romanize_latin(chunk)), sep))

    units: list[Unit] = []
    pending = ""
    for u, sep in raw:
        if u.roman:
            u.text = pending + (sep if pending else "") + u.text
            pending = ""
            units.append(u)
        elif units:
            units[-1].text += sep + u.text
        else:
            pending += (sep if pending else "") + u.text
    return units


_SECTION_RE = re.compile(r"^\s*\[[^\]]*\]\s*$")


def parse_lyrics(raw: str) -> list[str]:
    """Lyrics file -> list of sung lines. Blank lines and [Chorus]-style headers are removed.

    A line that is only "(yeah)" is kept: parentheses conventionally mark sung ad-libs.
    """
    lines = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln or _SECTION_RE.match(ln):
            continue
        lines.append(ln)
    return lines


def build_lines(texts: list[str]) -> list[Line]:
    lines = [Line(tokenize_line(t), t) for t in texts]
    return [ln for ln in lines if ln.units]


# --------------------------------------------------------------------------- CTC Viterbi

def ctc_align(log_probs: np.ndarray, targets: list[int], blank: int):
    """Best monotonic CTC alignment of `targets` to `log_probs` [T, V].

    Returns per-target (first_frame, end_frame_exclusive, mean_log_prob) arrays.
    """
    T = log_probs.shape[0]
    L = len(targets)
    if L == 0:
        raise AlignmentError("nothing to align")
    tg = np.asarray(targets, dtype=np.int64)
    repeats = int((tg[1:] == tg[:-1]).sum()) if L > 1 else 0
    if T < L + repeats:
        raise AlignmentError(f"audio too short for the text: {T} frames for {L} characters")
    S = 2 * L + 1
    if T * S > 900_000_000:
        raise AlignmentError(f"alignment too large ({T} frames x {S} states); try aligning shorter pieces")

    ext = np.full(S, blank, dtype=np.int64)
    ext[1::2] = tg
    can_skip = np.zeros(S, dtype=bool)  # skipping the blank between two *different* labels
    if L > 1:
        can_skip[3::2] = tg[1:] != tg[:-1]

    lp = np.ascontiguousarray(log_probs, dtype=np.float32)
    alpha = np.full(S, _NEG, dtype=np.float32)
    alpha[0] = lp[0, blank]
    alpha[1] = lp[0, ext[1]]
    back = np.zeros((T, S), dtype=np.uint8)
    prev1 = np.empty(S, dtype=np.float32)
    prev2 = np.empty(S, dtype=np.float32)
    for t in range(1, T):
        prev1[0] = _NEG
        prev1[1:] = alpha[:-1]
        prev2[:2] = _NEG
        prev2[2:] = np.where(can_skip[2:], alpha[:-2], _NEG)
        best = alpha
        step = np.zeros(S, dtype=np.uint8)
        m = prev1 > best
        best = np.where(m, prev1, best)
        step[m] = 1
        m = prev2 > best
        best = np.where(m, prev2, best)
        step[m] = 2
        back[t] = step
        alpha = best + lp[t, ext]

    state = S - 1 if S == 1 or alpha[S - 1] >= alpha[S - 2] else S - 2
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = state
        if t > 0:
            state -= int(back[t, state])
    if state not in (0, 1):
        raise AlignmentError("backtracking failed")  # defensive: cannot happen with a valid lattice

    frame_lp = lp[np.arange(T), ext[path]]
    states, first, counts = np.unique(path, return_index=True, return_counts=True)
    sums = np.add.reduceat(frame_lp, first)
    odd = states % 2 == 1
    tok = (states[odd] - 1) // 2
    if len(tok) != L or not np.array_equal(tok, np.arange(L)):
        raise AlignmentError("path skipped a character")  # defensive
    starts = first[odd]
    ends = starts + counts[odd]
    scores = sums[odd] / counts[odd]
    return starts, ends, scores


# --------------------------------------------------------------------------- emissions

def compute_emissions(wave: np.ndarray, infer: Callable[[np.ndarray], np.ndarray], window_s: int = 30,
                      context_s: int = 2, normalize: bool = True) -> np.ndarray:
    """Per-frame log-probabilities [T, V] for a whole recording, computed in overlapping windows.

    `infer(chunk)` maps a 16 kHz float32 chunk to log-probs [frames, V] at 320-sample stride. Each window
    gets `context_s` of real audio on both sides which is discarded, so window seams are invisible.
    """
    wave = np.asarray(wave, dtype=np.float32)
    if normalize:
        wave = (wave - wave.mean()) / (wave.std() + 1e-7)
    n = len(wave)
    win, ctx = window_s * SR, context_s * SR
    pad_end = (-n) % win
    padded = np.pad(wave, (ctx, ctx + pad_end))
    f0 = int(round(context_s / STRIDE_S))
    f1 = f0 + int(round(window_s / STRIDE_S))
    outs = []
    for start in range(0, n + pad_end, win):
        chunk = padded[start: start + win + 2 * ctx]
        lp = infer(chunk)
        outs.append(lp[f0:f1])
    em = np.concatenate(outs, axis=0)
    return em[: int(math.ceil(n / 320))]


@dataclass
class Aligner:
    infer: Callable[[np.ndarray], np.ndarray]
    vocab: dict[str, int]
    blank: int
    model: object = None
    description: str = ""


def _load_vocab(model_id: str, auto_tokenizer) -> tuple[dict[str, int], int | None]:
    """Character vocabulary and pad id. Reads vocab.json directly if the tokenizer class cannot be loaded."""
    try:
        tok = auto_tokenizer.from_pretrained(model_id)
        return dict(tok.get_vocab()), tok.pad_token_id
    except Exception:
        import json

        from huggingface_hub import hf_hub_download

        with open(hf_hub_download(model_id, "vocab.json"), encoding="utf-8") as fh:
            vocab = json.load(fh)
        return vocab, vocab.get("<pad>")


def load_aligner(model_id: str, device: str) -> Aligner:
    import torch
    from transformers import AutoModelForCTC, AutoTokenizer

    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        model = AutoModelForCTC.from_pretrained(model_id, dtype=dtype)
    except TypeError:  # older transformers spell it torch_dtype
        model = AutoModelForCTC.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device).eval()
    raw_vocab, pad_id = _load_vocab(model_id, AutoTokenizer)
    vocab = {k.lower(): v for k, v in raw_vocab.items()}
    blank = raw_vocab.get("<blank>", pad_id)
    if blank is None:
        raise AlignmentError(f"cannot find the CTC blank token in {model_id}")
    letters = sum(1 for c in "abcdefghijklmnopqrstuvwxyz'" if c in vocab)
    if letters < 20:
        sample = sorted(raw_vocab)[:40]
        raise AlignmentError(f"{model_id} does not look like a character-level aligner "
                             f"(only {letters}/27 letters in its vocabulary; sample: {sample})")

    @torch.inference_mode()
    def infer(chunk: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(chunk)[None].to(device=device, dtype=dtype)
        logits = model(x).logits.float()
        return torch.log_softmax(logits, dim=-1)[0].cpu().numpy()

    return Aligner(infer, vocab, blank, model, f"{model_id} (blank id {blank}, {letters}/27 letters in vocab)")


# --------------------------------------------------------------------------- units <-> frames

def align_units(emissions: np.ndarray, units: list[Unit], aligner: Aligner, frame_offset: int = 0) -> int:
    """Place `units` on the time axis (in place). `emissions` row 0 is global frame `frame_offset`.

    Returns the number of characters that were not in the model vocabulary and so were skipped.
    """
    targets: list[int] = []
    owner: list[int] = []  # which unit each target belongs to
    dropped = 0
    for i, u in enumerate(units):
        for ch in u.roman:
            tid = aligner.vocab.get(ch)
            if tid is None:
                dropped += 1
                continue
            targets.append(tid)
            owner.append(i)
    if not targets:
        raise AlignmentError("no alignable characters")
    starts, ends, scores = ctc_align(emissions, targets, aligner.blank)
    owner_arr = np.asarray(owner)
    for i, u in enumerate(units):
        idx = np.flatnonzero(owner_arr == i)
        if len(idx) == 0:
            continue
        u.start = (frame_offset + int(starts[idx[0]])) * STRIDE_S
        u.end = (frame_offset + int(ends[idx[-1]])) * STRIDE_S
        u.score = float(scores[idx].mean())
    return dropped


def spread_evenly(units: list[Unit], start: float, end: float) -> None:
    """Fallback when alignment is impossible: share the time by text length, mark score as unknown."""
    weights = np.array([max(1, len(u.roman)) for u in units], dtype=float)
    edges = start + (end - start) * np.concatenate([[0.0], np.cumsum(weights) / weights.sum()])
    for u, a, b in zip(units, edges[:-1], edges[1:]):
        u.start, u.end, u.score = float(a), float(b), None


def fill_gaps(units: list[Unit]) -> None:
    """Give units that never got a time (no alignable characters) a zero-length slot next to a neighbour."""
    last_end = None
    for u in units:
        if u.start is None and last_end is not None:
            u.start = u.end = last_end
        elif u.start is not None:
            last_end = u.end
    nxt = None
    for u in reversed(units):
        if u.start is None and nxt is not None:
            u.start = u.end = nxt
        elif u.start is not None:
            nxt = u.start


def align_lyrics(emissions: np.ndarray, lines: list[Line], aligner: Aligner) -> int:
    """Known lyrics: one global alignment over the whole recording."""
    flat = [u for ln in lines for u in ln.units]
    dropped = align_units(emissions, flat, aligner, 0)
    fill_gaps(flat)
    return dropped


def align_draft(emissions: np.ndarray, segs, aligner: Aligner, pad: float = 0.4):
    """ASR draft: align each sung region's text inside its own time window so errors cannot propagate."""
    groups: dict[tuple[float, float], list] = {}
    for s in segs:
        key = (s.slice_start, s.slice_end) if s.slice_start is not None else (s.start, s.end)
        groups.setdefault(key, []).append(s)

    lines: list[Line] = []
    failures = 0
    for (a, b), group in groups.items():
        per_line = [tokenize_line(s.text) for s in group]
        flat = [u for units in per_line for u in units]
        if not flat:
            continue
        f0 = max(0, int((a - pad) / STRIDE_S))
        f1 = min(len(emissions), int(math.ceil((b + pad) / STRIDE_S)))
        try:
            align_units(emissions[f0:f1], flat, aligner, f0)
            fill_gaps(flat)
        except AlignmentError:
            failures += 1
            spread_evenly(flat, a, b)
        for units, s in zip(per_line, group):
            if units:
                lines.append(Line(units, s.text))
    return lines, failures
