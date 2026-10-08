"""Whisper transcription (faster-whisper) tuned for songs: sliced input, no text conditioning, guards."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .util import log, preload_torch

SR = 16000

# Phrases Whisper tends to invent on music / silence.
HALLUCINATIONS = [
    r"thank(s| you)( so much)? for (watching|listening)", r"please subscribe", r"subtitles? by",
    r"amara\.org", r"ご視聴ありがとう", r"チャンネル登録", r"字幕", r"^\W*(music|音楽)\W*$", r"^[\W_♪♫]+$",
]
_HALLUC_RE = re.compile("|".join(HALLUCINATIONS), re.IGNORECASE)


@dataclass
class Seg:
    start: float
    end: float
    text: str
    lang: str | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    slice_start: float | None = None
    slice_end: float | None = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("start", "end", "text", "lang", "avg_logprob", "no_speech_prob", "slice_start", "slice_end", "flags")}


@dataclass
class LangPolicy:
    mode: str  # "fixed" | "restricted" | "auto"
    langs: list[str]


def parse_lang(spec: str) -> LangPolicy:
    items = [x.strip().lower() for x in spec.split(",") if x.strip()]
    if not items or items == ["auto"]:
        return LangPolicy("auto", [])
    return LangPolicy("fixed" if len(items) == 1 else "restricted", items)


UNCERTAIN_LOGPROB = -1.0   # Whisper's own "not confident" line
UNCERTAIN_NO_SPEECH = 0.6  # Whisper's own "does not sound like speech" line


@dataclass
class AsrOptions:
    beam: int = 3
    prompt: str | None = None
    temperature: tuple = (0.0, 0.2, 0.4, 0.6)
    # Whisper silently SKIPS an audio window when it thinks "not speech" (> 0.6) and is unsure (logprob <= -1.0).
    # Sung / pitched / processed voices trip that easily, so for sung regions the guards are off by default and
    # doubtful lines are kept and flagged "uncertain" instead of vanishing.
    guards: bool = False
    # Extra hypotheses: transcribe each region also pitch-shifted by these many semitones, keep the most confident.
    pitch_shifts: tuple = (0,)


@dataclass
class Region:
    """Diagnostics for one sung region: what Whisper heard and how sure it was."""
    start: float
    end: float
    lang: str | None = None
    shift: int = 0
    n_segments: int = 0
    avg_logprob: float | None = None
    max_no_speech: float | None = None
    text: str = ""
    candidates: dict = field(default_factory=dict)  # semitone shift -> confidence (None = produced no text)

    @property
    def status(self) -> str:
        if not self.text.strip():
            return "EMPTY"
        if (self.avg_logprob is not None and self.avg_logprob < UNCERTAIN_LOGPROB) or \
                (self.max_no_speech is not None and self.max_no_speech > UNCERTAIN_NO_SPEECH):
            return "uncertain"
        return "ok"


def shift_pitch(audio: np.ndarray, semitones: float, sr: int = SR) -> np.ndarray:
    """Pitch-shift without changing duration (a negative value lowers the voice)."""
    if not semitones:
        return audio
    import librosa

    return librosa.effects.pitch_shift(audio.astype(np.float32), sr=sr, n_steps=float(semitones)).astype(np.float32)


def load_model(name: str, device: str, compute_type: str | None, download_root: str | None = None):
    preload_torch()  # must happen before ctranslate2 loads CUDA libs on Windows
    from faster_whisper import WhisperModel

    if compute_type is None:
        compute_type = "int8_float16" if device == "cuda" else "int8"
    log(f"loading Whisper '{name}' on {device} ({compute_type}); first run downloads it")
    return WhisperModel(name, device=device, compute_type=compute_type, download_root=download_root)


def choose_language(model, audio: np.ndarray, policy: LangPolicy) -> str | None:
    if policy.mode == "fixed":
        return policy.langs[0]
    lang, _prob, all_probs = model.detect_language(audio=audio)
    if policy.mode == "auto":
        return lang
    probs = dict(all_probs)
    return max(policy.langs, key=lambda code: probs.get(code, 0.0))


def _run(model, audio: np.ndarray, lang: str | None, opts: AsrOptions, multilingual: bool = False,
         guards: bool = True):
    return model.transcribe(
        audio, language=lang, beam_size=opts.beam, best_of=3, temperature=list(opts.temperature),
        condition_on_previous_text=False, compression_ratio_threshold=2.4,
        log_prob_threshold=UNCERTAIN_LOGPROB if guards else None,
        no_speech_threshold=UNCERTAIN_NO_SPEECH if guards else None,
        initial_prompt=opts.prompt, vad_filter=False, word_timestamps=False, multilingual=multilingual,
    )


def _make_seg(s, start_offset: float, lang, slice_range=None) -> Seg:
    seg = Seg(start_offset + s.start, start_offset + s.end, s.text.strip(), lang, s.avg_logprob, s.no_speech_prob)
    if slice_range:
        seg.slice_start, seg.slice_end = slice_range
    if (s.avg_logprob is not None and s.avg_logprob < UNCERTAIN_LOGPROB) or \
            (s.no_speech_prob is not None and s.no_speech_prob > UNCERTAIN_NO_SPEECH):
        seg.flags.append("uncertain")
    return seg


def transcribe_full(model, audio: np.ndarray, policy: LangPolicy, opts: AsrOptions) -> list[Seg]:
    """Baseline: hand the whole file to Whisper in one go, with Whisper's normal skip guards."""
    if policy.mode == "fixed":
        lang, multilingual = policy.langs[0], False
    else:
        lang, multilingual = None, True
    segments, info = _run(model, audio, lang, opts, multilingual, guards=True)
    return [_make_seg(s, 0.0, lang or info.language) for s in segments]


def _confidence(segs: list[Seg]) -> float | None:
    """Duration-weighted mean log-probability of a hypothesis; None when it produced no text."""
    segs = [s for s in segs if s.text and s.avg_logprob is not None]
    if not segs:
        return None
    weights = np.array([max(s.end - s.start, 0.05) for s in segs])
    return float(np.average([s.avg_logprob for s in segs], weights=weights))


def transcribe_slices(model, audio: np.ndarray, slices: list[tuple[float, float]], policy: LangPolicy,
                      opts: AsrOptions) -> tuple[list[Seg], list[Region]]:
    """Transcribe each sung region separately: no context bleeds between regions, so no repetition loops.

    Returns (segments, one Region diagnostic per region). With several `opts.pitch_shifts` every region is
    transcribed at each shift and the most confident hypothesis wins.
    """
    out: list[Seg] = []
    regions: list[Region] = []
    for i, (a, b) in enumerate(slices, 1):
        chunk = audio[int(a * SR): int(b * SR)]
        if len(chunk) < int(0.3 * SR):
            continue
        lang = choose_language(model, chunk, policy)
        best = None  # (rank, shift, segs)
        candidates: dict = {}
        for shift in opts.pitch_shifts:
            segments, _info = _run(model, shift_pitch(chunk, shift), lang, opts, guards=opts.guards)
            segs = [_make_seg(s, a, lang, (a, b)) for s in segments if s.text.strip()]
            conf = _confidence(segs)
            candidates[shift] = conf
            rank = float("-inf") if conf is None else conf
            if best is None or rank > best[0]:
                best = (rank, shift, segs)
        _rank, shift, segs = best
        out.extend(segs)
        probs = [s.no_speech_prob for s in segs if s.no_speech_prob is not None]
        regions.append(Region(a, b, lang, shift, len(segs), candidates[shift], max(probs) if probs else None,
                              " ".join(s.text for s in segs), candidates))
        if i % 10 == 0 or i == len(slices):
            log(f"  transcribed {i}/{len(slices)} regions")
    return out, regions


def filter_segments(segs: list[Seg], drop_silence: bool = False) -> tuple[list[Seg], list[tuple[Seg, str]]]:
    """Drop likely hallucinations and runaway repeats. Returns (kept, [(dropped, reason)]).

    Doubtful-but-real lines are kept (flagged "uncertain"); `drop_silence=True` restores Whisper-style removal.
    """
    kept: list[Seg] = []
    dropped: list[tuple[Seg, str]] = []
    run_text, run_len = None, 0
    for s in segs:
        text = s.text.strip()
        if not text:
            dropped.append((s, "empty"))
            continue
        if _HALLUC_RE.search(text):
            dropped.append((s, "known hallucination phrase"))
            continue
        if drop_silence and s.no_speech_prob is not None and s.avg_logprob is not None \
                and s.no_speech_prob > 0.8 and s.avg_logprob < UNCERTAIN_LOGPROB:
            dropped.append((s, "looks like silence"))
            continue
        norm = re.sub(r"\W+", "", text.lower())
        run_len = run_len + 1 if norm == run_text else 1
        run_text = norm
        if run_len > 3:
            dropped.append((s, "repeated >3x in a row"))
            continue
        kept.append(s)
    return kept, dropped
