"""Every keyword we pass to faster-whisper must exist in the installed version (weights are not needed)."""
import inspect
import types

import numpy as np
import pytest

faster_whisper = pytest.importorskip("faster_whisper")

from lyricspike import asr  # noqa: E402


class Recorder:
    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        return iter([]), types.SimpleNamespace(language="en")


def test_transcribe_kwargs_match_installed_faster_whisper():
    rec = Recorder()
    audio = np.zeros(16000, dtype=np.float32)
    asr.transcribe_full(rec, audio, asr.parse_lang("en,ja"), asr.AsrOptions(prompt="x"))
    full_kwargs = rec.kwargs
    asr.transcribe_slices(rec, audio, [(0.0, 0.9)], asr.parse_lang("ja"), asr.AsrOptions())
    slice_kwargs = rec.kwargs
    sig = inspect.signature(faster_whisper.WhisperModel.transcribe)
    for kwargs in (full_kwargs, slice_kwargs):
        sig.bind(object(), audio, **kwargs)           # raises TypeError on any unknown/misspelled keyword
    assert slice_kwargs["condition_on_previous_text"] is False
    assert full_kwargs["multilingual"] is True and full_kwargs["language"] is None
    assert slice_kwargs["language"] == "ja" and slice_kwargs["multilingual"] is False


def test_detect_language_call_matches():
    sig = inspect.signature(faster_whisper.WhisperModel.detect_language)
    sig.bind(object(), audio=np.zeros(10, dtype=np.float32))
    sig2 = inspect.signature(faster_whisper.WhisperModel.__init__)
    sig2.bind(object(), "large-v2", device="cuda", compute_type="int8_float16", download_root=None)
