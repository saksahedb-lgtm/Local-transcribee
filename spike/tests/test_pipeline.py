"""Whole-pipeline wiring test with the heavy models replaced by stand-ins (no GPU, no downloads)."""
import json
import types

import numpy as np
import pytest
import soundfile as sf

from lyricspike import align, asr, pipeline, separate
from lyricspike.align import Aligner


def make_song(path, seconds=14, sr=44100):
    rng = np.random.default_rng(1)
    wave = np.zeros((int(seconds * sr), 2), dtype=np.float32)
    for a, b in [(1.0, 4.0), (6.0, 9.5), (11.0, 13.0)]:
        wave[int(a * sr): int(b * sr)] = rng.standard_normal((int((b - a) * sr), 2)) * 0.2
    sf.write(path, wave, sr)


class FakeModel:
    calls = 0

    def transcribe(self, audio, language=None, **kw):
        FakeModel.calls += 1
        seg = types.SimpleNamespace(start=0.1, end=min(2.0, len(audio) / 16000), text=f" la la {language} ",
                                    avg_logprob=-0.4, no_speech_prob=0.05)
        return iter([seg]), types.SimpleNamespace(language=language or "en")

    def detect_language(self, audio=None):
        return "en", 0.7, [("en", 0.7), ("ja", 0.2)]


def fake_aligner(*_a, **_k):
    vocab = {c: i + 1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz'")}
    rng = np.random.default_rng(0)

    def infer(chunk):
        frames = (len(chunk) - 400) // 320 + 1
        logits = rng.standard_normal((frames, len(vocab) + 1)).astype(np.float32)
        logits[:, 0] += 3  # blank-heavy, like a real CTC model
        return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))

    return Aligner(infer=infer, vocab=vocab, blank=0, description="fake aligner")


@pytest.fixture
def patched(monkeypatch, tmp_path):
    def fake_sep(input_wav, stems_dir, models, *a, **k):
        target = stems_dir / "vocals.wav"
        data, sr = sf.read(input_wav)
        sf.write(target, data, sr)
        return target

    monkeypatch.setattr(separate, "separate_vocals", fake_sep)
    monkeypatch.setattr(asr, "load_model", lambda *a, **k: FakeModel())
    monkeypatch.setattr(align, "load_aligner", fake_aligner)
    song = tmp_path / "My Song.flac"
    make_song(song)
    return song


def args_for(song, tmp_path, **over):
    base = dict(song=str(song), out=str(tmp_path / "out"), lyrics=None, reference=None, lang="en,ja",
                device="cpu", force=False, no_separate=False, sep_model=None, ensemble=False,
                karaoke_model=None, sep_precision="fp16", sep_segment=None, sep_overlap=None, model_dir=None,
                no_asr=False, asr_model="tiny", asr_compute=None, beam=1, prompt=None, variants=None,
                no_align=False, align_from=None, align_on="vocals", aligner_model="x")
    base.update(over)
    return types.SimpleNamespace(**base)


def test_draft_mode_with_reference(patched, tmp_path):
    ref = tmp_path / "ref.txt"
    ref.write_text("la la en\nla la en\n", encoding="utf-8")
    report = pipeline.run(args_for(patched, tmp_path, reference=str(ref), align_from="vocals"))
    out = report.parent
    text = report.read_text(encoding="utf-8")
    assert "Transcription variants" in text and "mix_raw" in text and "| vocals |" in text
    assert "WER" in text and "Alignment" in text
    assert (out / "asr" / "vocals.txt").read_text(encoding="utf-8").strip()
    assert (out / "original.flac").exists()
    lrc = (out / "align" / "lyrics.lrc").read_text(encoding="utf-8").splitlines()
    assert len(lrc) >= 3 and all(l.startswith("[") for l in lrc)
    data = json.loads((out / "align" / "aligned.json").read_text(encoding="utf-8"))
    starts = [ln["start"] for ln in data["lines"]]
    assert starts == sorted(starts)
    viewer = (out / "viewer.html").read_text(encoding="utf-8")
    assert "stems/vocals.wav" in viewer and "/*__PAYLOAD__*/null" not in viewer


def test_pasted_lyrics_mode_without_asr(patched, tmp_path):
    lyr = tmp_path / "lyrics.txt"
    lyr.write_text("[Verse]\nlet it go\nlet it go\n\n夜に溶けていく\n", encoding="utf-8")
    FakeModel.calls = 0
    report = pipeline.run(args_for(patched, tmp_path, lyrics=str(lyr), no_asr=True))
    assert FakeModel.calls == 0
    out = report.parent
    lrc = (out / "align" / "lyrics.lrc").read_text(encoding="utf-8").splitlines()
    assert len(lrc) == 3 and lrc[2].endswith("夜に溶けていく")
    assert "pasted lyrics" in report.read_text(encoding="utf-8")


def test_failures_are_reported_not_fatal(patched, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory (simulated)")

    monkeypatch.setattr(separate, "separate_vocals", boom)
    report = pipeline.run(args_for(patched, tmp_path, no_align=True))
    text = report.read_text(encoding="utf-8")
    assert "Problems" in text and "out of memory" in text and "--sep-segment" in text
    assert "mix_raw" in text      # the baseline still ran
