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


@dataclass
class AsrOptions:
    beam: int = 3
    prompt: str | None = None
    temperature: tuple = (0.0, 0.2, 0.4, 0.6)


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


def _run(model, audio: np.ndarray, lang: str | None, opts: AsrOptions, multilingual: bool = False):
    return model.transcribe(
        audio, language=lang, beam_size=opts.beam, best_of=3, temperature=list(opts.temperature),
        condition_on_previous_text=False, compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
        no_speech_threshold=0.6, initial_prompt=opts.prompt, vad_filter=False, word_timestamps=False,
        multilingual=multilingual,
    )


def transcribe_full(model, audio: np.ndarray, policy: LangPolicy, opts: AsrOptions) -> list[Seg]:
    """Baseline: hand the whole file to Whisper in one go."""
    if policy.mode == "fixed":
        lang, multilingual = policy.langs[0], False
    else:
        lang, multilingual = None, True
    segments, info = _run(model, audio, lang, opts, multilingual)
    return [Seg(s.start, s.end, s.text.strip(), lang or info.language, s.avg_logprob, s.no_speech_prob)
            for s in segments]


def transcribe_slices(model, audio: np.ndarray, slices: list[tuple[float, float]], policy: LangPolicy,
                      opts: AsrOptions) -> list[Seg]:
    """Transcribe each sung region separately: no context bleeds between regions, so no repetition loops."""
    out: list[Seg] = []
    for i, (a, b) in enumerate(slices, 1):
        chunk = audio[int(a * SR): int(b * SR)]
        if len(chunk) < int(0.3 * SR):
            continue
        lang = choose_language(model, chunk, policy)
        segments, _info = _run(model, chunk, lang, opts)
        for s in segments:
            out.append(Seg(a + s.start, a + s.end, s.text.strip(), lang, s.avg_logprob, s.no_speech_prob,
                           slice_start=a, slice_end=b))
        if i % 10 == 0 or i == len(slices):
            log(f"  transcribed {i}/{len(slices)} regions")
    return out


def filter_segments(segs: list[Seg]) -> tuple[list[Seg], list[tuple[Seg, str]]]:
    """Drop likely hallucinations and runaway repeats. Returns (kept, [(dropped, reason)])."""
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
        if s.no_speech_prob is not None and s.avg_logprob is not None \
                and s.no_speech_prob > 0.8 and s.avg_logprob < -1.0:
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
