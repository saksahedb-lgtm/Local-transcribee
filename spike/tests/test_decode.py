import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from lyricspike import audio

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed to make test files")


@pytest.mark.parametrize("ext,codec", [("mp3", []), ("flac", []), ("ogg", []), ("m4a", []), ("wav", [])])
def test_decode_formats(tmp_path, ext, codec):
    sr = 44100
    t = np.arange(sr * 3) / sr
    stereo = np.stack([0.4 * np.sin(2 * np.pi * 440 * t), 0.4 * np.sin(2 * np.pi * 880 * t)], axis=1).astype(np.float32)
    src = tmp_path / "src.wav"
    sf.write(src, stereo, sr)
    dst = tmp_path / f"song.{ext}"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), *codec, str(dst)], check=True)

    st = audio.decode(dst, 44100, stereo=True)
    assert st.ndim == 2 and st.shape[1] == 2
    assert abs(len(st) / 44100 - 3.0) < 0.1
    left_freq = np.argmax(np.abs(np.fft.rfft(st[:, 0]))) * 44100 / len(st)
    right_freq = np.argmax(np.abs(np.fft.rfft(st[:, 1]))) * 44100 / len(st)
    assert abs(left_freq - 440) < 10 and abs(right_freq - 880) < 10     # channels kept apart

    mono = audio.decode(dst, 16000)
    assert mono.ndim == 1 and abs(len(mono) / 16000 - 3.0) < 0.1
    assert np.abs(mono).max() <= 1.0 + 1e-3


def test_decode_mono_source_as_stereo(tmp_path):
    sr = 22050
    sf.write(tmp_path / "m.wav", (0.3 * np.sin(np.arange(sr * 2) / 20)).astype(np.float32), sr)
    st = audio.decode(tmp_path / "m.wav", 44100, stereo=True)
    assert st.shape[1] == 2 and np.allclose(st[:, 0], st[:, 1], atol=1e-4)


def test_decode_rejects_non_audio(tmp_path):
    bad = tmp_path / "x.mp3"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(Exception):
        audio.decode(bad, 16000)
