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
    # sung regions must NOT be silently skipped by Whisper's "not speech / unsure" guards...
    assert slice_kwargs["no_speech_threshold"] is None and slice_kwargs["log_prob_threshold"] is None
    # ...while the whole-file baseline keeps Whisper's normal behaviour
    assert full_kwargs["no_speech_threshold"] == 0.6 and full_kwargs["log_prob_threshold"] == -1.0
    assert full_kwargs["multilingual"] is True and full_kwargs["language"] is None
    assert slice_kwargs["language"] == "ja" and slice_kwargs["multilingual"] is False


def test_detect_language_call_matches():
    sig = inspect.signature(faster_whisper.WhisperModel.detect_language)
    sig.bind(object(), audio=np.zeros(10, dtype=np.float32))
    sig2 = inspect.signature(faster_whisper.WhisperModel.__init__)
    sig2.bind(object(), "large-v2", device="cuda", compute_type="int8_float16", download_root=None)


def test_guards_can_be_turned_back_on():
    rec = Recorder()
    audio = np.zeros(16000, dtype=np.float32)
    asr.transcribe_slices(rec, audio, [(0.0, 0.9)], asr.parse_lang("en"), asr.AsrOptions(guards=True))
    assert rec.kwargs["no_speech_threshold"] == 0.6 and rec.kwargs["log_prob_threshold"] == -1.0


def test_pitch_shift_hypotheses_pick_the_most_confident(monkeypatch):
    """A model that is more sure of the lowered voice should win; regions record every candidate."""
    seen = []

    class PitchSensitive:
        def detect_language(self, audio=None):
            return "en", 0.9, [("en", 0.9)]

        def transcribe(self, audio, **kw):
            seen.append(float(audio[0]))
            lowered = audio[0] < 0.5                      # the (fake) shift marks audio by changing its first sample
            seg = types.SimpleNamespace(start=0.0, end=0.8, text=" low " if lowered else " high ",
                                        avg_logprob=-0.4 if lowered else -1.6, no_speech_prob=0.1)
            return iter([seg]), types.SimpleNamespace(language="en")

    monkeypatch.setattr(asr, "shift_pitch", lambda a, s, sr=16000: (a * 0 + (0.1 if s else 1.0)).astype(np.float32))
    audio = np.ones(16000, dtype=np.float32)
    segs, regions = asr.transcribe_slices(PitchSensitive(), audio, [(0.0, 0.9)], asr.parse_lang("en"),
                                          asr.AsrOptions(pitch_shifts=(0, -4)))
    assert [s.text for s in segs] == ["low"]
    r = regions[0]
    assert r.shift == -4 and r.status == "ok"
    assert r.candidates[0] == pytest.approx(-1.6) and r.candidates[-4] == pytest.approx(-0.4)


def test_empty_and_uncertain_status():
    r = asr.Region(0, 1, text="", candidates={0: None})
    assert r.status == "EMPTY"
    assert asr.Region(0, 1, text="x", avg_logprob=-1.4, max_no_speech=0.1).status == "uncertain"
    assert asr.Region(0, 1, text="x", avg_logprob=-0.3, max_no_speech=0.9).status == "uncertain"
    assert asr.Region(0, 1, text="x", avg_logprob=-0.3, max_no_speech=0.1).status == "ok"
