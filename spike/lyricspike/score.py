"""Compare a transcript with reference lyrics: WER, CER and a phonetic CER that is tolerant of kanji/kana."""
from __future__ import annotations

import re
import unicodedata

from .align import romanize_text


def normalize(text: str) -> str:
    """Lowercase, drop [Section] headers and punctuation, keep letters/numbers (incl. Japanese)."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = text.replace("'", "").replace("’", "")
    text = "".join(ch if unicodedata.category(ch)[0] in "LNM" or ch.isspace() else " " for ch in text)
    return " ".join(text.split())


def _rate(fn, ref: str, hyp: str) -> float | None:
    if not ref:
        return None
    if not hyp:
        return 1.0
    return float(fn(ref, hyp))


def evaluate(reference: str, hypothesis: str) -> dict:
    """WER (word level; meaningless for Japanese), CER (characters) and pCER (romanized characters)."""
    import jiwer

    ref, hyp = normalize(reference), normalize(hypothesis)
    ref_c, hyp_c = ref.replace(" ", ""), hyp.replace(" ", "")
    ref_p, hyp_p = romanize_text(ref), romanize_text(hyp)
    return {
        "wer": _rate(jiwer.wer, ref, hyp),
        "cer": _rate(jiwer.cer, ref_c, hyp_c),
        "pcer": _rate(jiwer.cer, ref_p, hyp_p),
        "ref_words": len(ref.split()),
        "ref_chars": len(ref_c),
        "hyp_words": len(hyp.split()),
    }
