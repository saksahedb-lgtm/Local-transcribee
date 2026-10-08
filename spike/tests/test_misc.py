import numpy as np
import pytest

from lyricspike import asr, audio, export, score
from lyricspike.align import Line, Unit


def burst(sr, seconds, amp=0.3, seed=0):
    return (np.random.default_rng(seed).standard_normal(int(seconds * sr)) * amp).astype(np.float32)


def test_vocal_segments_merge_drop_split():
    sr = 16000
    parts = [(2.0, 4.0), (4.4, 6.0), (10.0, 12.5), (15.0, 15.1), (20.0, 60.0)]
    wave = np.zeros(int(65 * sr), dtype=np.float32)
    for i, (a, b) in enumerate(parts):
        lo, hi = int(a * sr), int(b * sr)
        wave[lo:hi] = burst(sr, (hi - lo) / sr, seed=i)[: hi - lo]
    segs = audio.vocal_segments(wave, sr)
    starts = [round(a) for a, _ in segs]
    assert starts[:2] == [2, 10]              # 2-4 and 4.4-6 merged (gap 0.4 s); blip at 15 s dropped
    assert 15 not in starts
    assert segs[0][1] == pytest.approx(6.2, abs=0.15)
    long_pieces = [s for s in segs if s[0] >= 19]
    assert len(long_pieces) >= 2               # 40 s burst split to fit Whisper's window
    assert all(b - a <= 26 for a, b in long_pieces)
    assert all(segs[i][1] <= segs[i + 1][0] for i in range(len(segs) - 1))


def test_vocal_segments_silence_and_tiny():
    assert audio.vocal_segments(np.zeros(16000 * 5, dtype=np.float32)) == []
    assert audio.vocal_segments(np.zeros(10, dtype=np.float32)) == []


def test_score_normalization_and_metrics():
    m = score.evaluate("[Chorus]\nI don't know, yeah!", "i dont know yeah")
    assert m["wer"] == 0 and m["cer"] == 0
    m = score.evaluate("hello world", "hello word")
    assert m["wer"] == pytest.approx(0.5)
    jp = score.evaluate("夜に溶けていく", "よるにとけていく")
    assert jp["cer"] > 0.3 and jp["pcer"] < 0.1     # kanji vs kana spelling: phonetic metric forgives it
    assert score.evaluate("a b", "")["wer"] == 1.0
    assert score.evaluate("", "x")["wer"] is None


def test_filter_segments():
    S = lambda t, lp=-0.3, ns=0.1: asr.Seg(0, 1, t, avg_logprob=lp, no_speech_prob=ns)
    segs = [S("real lyric"), S("Thanks for watching!"), S("la"), S("la"), S("la"), S("la"), S("la"),
            S("noise", lp=-1.5, ns=0.95), S("")]
    kept, dropped = asr.filter_segments(segs)
    assert [s.text for s in kept] == ["real lyric", "la", "la", "la"]
    reasons = [r for _, r in dropped]
    assert "known hallucination phrase" in reasons and "looks like silence" in reasons
    assert reasons.count("repeated >3x in a row") == 2


def test_language_policy():
    class M:
        def detect_language(self, audio=None):
            return "en", 0.6, [("en", 0.6), ("ja", 0.3), ("fr", 0.05)]

    assert asr.choose_language(M(), None, asr.parse_lang("ja")) == "ja"
    assert asr.choose_language(M(), None, asr.parse_lang("auto")) == "en"
    assert asr.choose_language(M(), None, asr.parse_lang("ja,fr")) == "ja"
    assert asr.parse_lang("en,ja").mode == "restricted"


def test_lrc_formats():
    assert export.lrc_ts(59.999) == "[01:00.00]"
    assert export.lrc_ts(75.5) == "[01:15.50]"
    line = Line([Unit("hi", "hi", 1.0, 1.4, -0.5), Unit("there", "there", 1.6, 2.0, -0.7)], "hi there")
    assert export.to_lrc([line]) == "[00:01.00]hi there\n"
    assert export.to_enhanced_lrc([line]) == "[00:01.00] <00:01.00>hi <00:01.60>there\n"
    d = export.lines_to_dict([line], "test")
    assert d["lines"][0]["words"][1]["start"] == 1.6


def test_viewer_embeds_payload_safely(tmp_path):
    line = Line([Unit("</script>", "script", 1.0, 1.4, -0.5)], "</script>")
    out = tmp_path / "v.html"
    export.write_viewer(out, export.lines_to_dict([line], "t"), [("x", "x.wav")], "title")
    html = out.read_text(encoding="utf-8")
    assert "/*__PAYLOAD__*/null" not in html
    assert html.count("</script>") == 1   # the lyric's own "</script>" was escaped


def test_karaoke_stem_classification(tmp_path):
    import soundfile as sf
    from lyricspike import separate

    loud, quiet = tmp_path / "loud.wav", tmp_path / "quiet.wav"
    sf.write(loud, burst(16000, 1, 0.5), 16000)
    sf.write(quiet, burst(16000, 1, 0.05), 16000)
    # a model whose target is the karaoke track: 'karaoke' = music+backing, residual 'other' = lead
    assert separate.classify_karaoke_stems({"karaoke": quiet, "other": loud})[:2] == ("other", "karaoke")
    assert separate.classify_karaoke_stems({"other": loud, "karaoke": quiet})[:2] == ("other", "karaoke")
    assert separate.classify_karaoke_stems({"vocals": loud, "instrumental": quiet})[:2] == ("vocals", "instrumental")
    # unknown names fall back to loudness
    lead, backing, how = separate.classify_karaoke_stems({"foo": quiet, "bar": loud})
    assert (lead, backing) == ("bar", "foo") and "loudness" in how


def test_ensure_ffmpeg_falls_back_to_bundled_binary(tmp_path, monkeypatch):
    import shutil
    import subprocess

    from lyricspike import util

    monkeypatch.setenv("PATH", "")
    assert shutil.which("ffmpeg") is None
    path = util.ensure_ffmpeg(cache_dir=tmp_path)
    assert str(tmp_path) in path
    assert subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0
    assert util.ensure_ffmpeg(cache_dir=tmp_path) == shutil.which("ffmpeg")   # second call: already on PATH


def test_load_vocab_falls_back_to_vocab_json(tmp_path, monkeypatch):
    import json

    import huggingface_hub

    from lyricspike import align

    vocab_file = tmp_path / "vocab.json"
    vocab_file.write_text(json.dumps({"<pad>": 0, "a": 1, "b": 2}), encoding="utf-8")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, name: str(vocab_file))

    class BrokenTokenizer:
        @staticmethod
        def from_pretrained(_):
            raise ValueError("tokenizer class not supported")

    vocab, pad = align._load_vocab("some/model", BrokenTokenizer)
    assert vocab["a"] == 1 and pad == 0

    class GoodTokenizer:
        pad_token_id = 7

        @staticmethod
        def from_pretrained(_):
            return GoodTokenizer()

        def get_vocab(self):
            return {"x": 1}

    assert align._load_vocab("m", GoodTokenizer) == ({"x": 1}, 7)
