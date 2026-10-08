import itertools

import numpy as np
import pytest

from lyricspike import align
from lyricspike.align import (Aligner, AlignmentError, Unit, build_lines, compute_emissions, ctc_align,
                              parse_lyrics, tokenize_line)

VOCAB = {c: i + 1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz'")}  # 0 = blank
V = len(VOCAB) + 1


def spiky(T, placements):
    """Log-probs where each (frame, token_id) is a confident peak and everything else is blank."""
    lp = np.full((T, V), np.log(1e-4), dtype=np.float32)
    lp[:, 0] = np.log(0.97)
    for f, tid in placements:
        lp[f, :] = np.log(1e-4)
        lp[f, tid] = np.log(0.97)
    return lp


def test_recovers_planted_positions():
    text = "hello world"
    ids = [VOCAB[c] for c in text.replace(" ", "")]
    frames = [3, 8, 12, 20, 21 + 6, 40, 47, 55, 60, 71]
    lp = spiky(80, list(zip(frames, ids)))
    starts, ends, scores = ctc_align(lp, ids, 0)
    assert list(starts) == frames
    assert (scores > np.log(0.5)).all()
    assert (ends > starts).all()


def test_repeated_letters_need_blank_between():
    ids = [VOCAB["l"], VOCAB["l"]]
    lp = spiky(10, [(2, ids[0]), (6, ids[1])])
    starts, _, _ = ctc_align(lp, ids, 0)
    assert list(starts) == [2, 6]


def test_too_short_raises():
    with pytest.raises(AlignmentError):
        ctc_align(spiky(3, []), [1, 2, 3, 4], 0)


def brute_force_best(lp, targets, blank):
    """Exhaustive CTC best-path score over all monotonic assignments (tiny sizes only)."""
    T, L = lp.shape[0], len(targets)
    best = -np.inf
    # a path is a non-decreasing state sequence over the 2L+1 extended states, steps of 0/1/2
    S = 2 * L + 1
    ext = [blank] * S
    for i, t in enumerate(targets):
        ext[2 * i + 1] = t

    def rec(t, s, acc):
        nonlocal best
        if t == T:
            if s in (S - 1, S - 2):
                best = max(best, acc)
            return
        for step in (0, 1, 2):
            ns = s + step
            if ns >= S:
                continue
            if step == 2 and not (ns % 2 == 1 and ns >= 3 and ext[ns] != ext[ns - 2]):
                continue
            rec(t + 1, ns, acc + lp[t, ext[ns]])

    for s0 in (0, 1):
        rec(1, s0, lp[0, ext[s0]])
    return best


def path_score(lp, targets, blank, starts, ends):
    T = lp.shape[0]
    ext_path = np.full(T, blank)
    for tok, (a, b) in enumerate(zip(starts, ends)):
        ext_path[a:b] = targets[tok]
    return float(lp[np.arange(T), ext_path].sum())


@pytest.mark.parametrize("seed", range(12))
def test_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    L = int(rng.integers(1, 4))
    targets = [int(x) for x in rng.integers(1, 4, size=L)]  # small alphabet => repeats happen
    repeats = sum(1 for i in range(1, L) if targets[i] == targets[i - 1])
    T = int(rng.integers(L + repeats, L + repeats + 5))
    lp = np.log(rng.dirichlet(np.ones(5), size=T)).astype(np.float32)
    starts, ends, _ = ctc_align(lp, targets, 0)
    assert np.isclose(path_score(lp, targets, 0, starts, ends), brute_force_best(lp, targets, 0), atol=1e-3)


def test_stitching_is_seamless():
    n = 16000 * 70 + 123  # more than two 30 s windows, ragged end

    def infer(chunk):  # frame j reports the waveform value at its first sample
        frames = (len(chunk) - 400) // 320 + 1
        return chunk[: frames * 320 : 320].reshape(-1, 1).astype(np.float32)

    wave = np.arange(n, dtype=np.float32)
    em = compute_emissions(wave, infer, normalize=False)
    assert em.shape[0] == int(np.ceil(n / 320))
    k = np.arange(em.shape[0])
    assert np.array_equal(em[:, 0], (k * 320).astype(np.float32))


def make_aligner():
    return Aligner(infer=None, vocab=VOCAB, blank=0)


def test_align_units_times_and_unknown_chars():
    units = tokenize_line("hi yo")
    ids = [VOCAB[c] for c in "hiyo"]
    frames = [10, 14, 30, 33]
    lp = spiky(60, list(zip(frames, ids)))
    dropped = align.align_units(lp, units, make_aligner(), frame_offset=100)
    assert dropped == 0
    assert units[0].start == pytest.approx((100 + 10) * 0.02)
    assert units[1].start == pytest.approx((100 + 30) * 0.02)
    assert units[1].end >= units[1].start


def test_tokenize_english_and_symbols():
    assert [u.text for u in tokenize_line("Don't stop - 808 ...")] == ["Don't", "stop -", "808 ..."]
    assert tokenize_line("808")[0].roman == "eightzeroeight"
    assert tokenize_line("café")[0].roman == "cafe"


def test_tokenize_punctuation_spacing_and_japanese_punct():
    assert [u.text for u in tokenize_line("- hello")] == ["- hello"]
    jp = tokenize_line("好き、 yeah")
    assert "".join(u.text for u in jp) == "好き、yeah" or [u.text for u in jp][0].endswith("、")


def test_tokenize_japanese():
    units = tokenize_line("夜に溶けていく")
    assert [u.roman for u in units] == ["yoru", "ni", "toke", "teiku"]
    assert "".join(u.text for u in units) == "夜に溶けていく"
    mixed = tokenize_line("love 愛してる")
    assert mixed[0].roman == "love" and mixed[1].roman == "itoshi"


def test_parse_lyrics_strips_headers_and_blanks():
    raw = "[Verse 1]\nfirst line\n\n(yeah)\nsecond (yeah) line\n[x2]\n"
    assert parse_lyrics(raw) == ["first line", "(yeah)", "second (yeah) line"]


def test_gap_filling_and_even_spread():
    units = [Unit("a", "a"), Unit("...", ""), Unit("b", "b")]
    units[0].start, units[0].end = 1.0, 1.5
    units[2].start, units[2].end = 3.0, 3.4
    align.fill_gaps(units)
    assert units[1].start == 1.5
    units2 = [Unit("ab", "ab"), Unit("cdef", "cdef")]
    align.spread_evenly(units2, 0.0, 6.0)
    assert units2[0].end == pytest.approx(2.0) and units2[1].end == pytest.approx(6.0)


def test_global_lyrics_alignment_end_to_end_on_synthetic_emissions():
    lines = build_lines(["go go", "stop"])
    flat = [u for ln in lines for u in ln.units]
    ids = [VOCAB[c] for u in flat for c in u.roman]
    frames = [5, 12, 25, 33, 60, 66, 71, 80]
    lp = spiky(100, list(zip(frames, ids)))
    align.align_lyrics(lp, lines, make_aligner())
    assert lines[0].start == pytest.approx(5 * 0.02)
    assert lines[1].start == pytest.approx(60 * 0.02)
    assert lines[0].end <= lines[1].start
