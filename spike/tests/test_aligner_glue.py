"""Exercise load_aligner/compute_emissions with real torch + transformers on a tiny random wav2vec2 CTC model.

Emissions from a random model are meaningless, so this only proves the plumbing: loading, vocab/blank
detection, frame stride, window stitching, and that alignment runs on the result.
"""
import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from lyricspike import align  # noqa: E402


def make_tiny_aligner_dir(tmp_path, with_blank_token=False):
    from transformers import Wav2Vec2Config, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC

    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    if with_blank_token:
        vocab = {"<blank>": 0, "<pad>": 1, "<s>": 2, "</s>": 3, "<unk>": 4}
    for ch in "abcdefghijklmnopqrstuvwxyz'":
        vocab[ch] = len(vocab)
    vocab["|"] = len(vocab)
    vocab_file = tmp_path / "vocab.json"
    vocab_file.write_text(json.dumps(vocab), encoding="utf-8")
    tok = Wav2Vec2CTCTokenizer(str(vocab_file), unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")
    cfg = Wav2Vec2Config(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=64, conv_dim=(16,) * 7, num_conv_pos_embeddings=16,
                         num_conv_pos_embedding_groups=2, pad_token_id=vocab["<pad>"])
    model = Wav2Vec2ForCTC(cfg)
    out = tmp_path / ("m_blank" if with_blank_token else "m")
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out, vocab


@pytest.mark.parametrize("with_blank_token", [False, True])
def test_load_aligner_and_emissions(tmp_path, with_blank_token):
    path, vocab = make_tiny_aligner_dir(tmp_path, with_blank_token)
    aligner = align.load_aligner(str(path), "cpu")
    assert aligner.blank == (vocab["<blank>"] if with_blank_token else vocab["<pad>"])
    assert aligner.vocab["a"] == vocab["a"] and "fake aligner" not in aligner.description

    chunk = np.random.default_rng(0).standard_normal(34 * 16000).astype(np.float32)
    lp = aligner.infer(chunk)
    assert lp.shape == ((len(chunk) - 400) // 320 + 1, len(vocab))        # 20 ms frames, as assumed
    assert np.allclose(np.exp(lp).sum(axis=1), 1.0, atol=1e-4)           # proper log-probabilities

    n = 16000 * 65 + 777
    wave = np.random.default_rng(1).standard_normal(n).astype(np.float32)
    em = align.compute_emissions(wave, aligner.infer)
    assert em.shape == (math.ceil(n / 320), len(vocab))

    lines = align.build_lines(["hello world", "let it go", "夜に溶けていく"])
    align.align_lyrics(em, lines, aligner)
    starts = [u.start for ln in lines for u in ln.units]
    assert all(s is not None for s in starts) and starts == sorted(starts)


def test_rejects_non_character_vocab(tmp_path):
    from transformers import Wav2Vec2Config, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC

    vocab = {"<pad>": 0, "<unk>": 1, "x": 2, "y": 3, "|": 4}
    vf = tmp_path / "vocab.json"
    vf.write_text(json.dumps(vocab), encoding="utf-8")
    tok = Wav2Vec2CTCTokenizer(str(vf), unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")
    cfg = Wav2Vec2Config(vocab_size=5, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=64, conv_dim=(16,) * 7, num_conv_pos_embeddings=16,
                         num_conv_pos_embedding_groups=2)
    out = tmp_path / "bad"
    Wav2Vec2ForCTC(cfg).save_pretrained(out)
    tok.save_pretrained(out)
    with pytest.raises(align.AlignmentError, match="character-level"):
        align.load_aligner(str(out), "cpu")
