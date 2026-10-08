#!/usr/bin/env python
"""lyricspike: feasibility spike for local lyrics transcription + sync of heavily produced songs.

  python spike.py doctor
  python spike.py run SONG.mp3 [--lyrics lyrics.txt] [--reference lyrics.txt] [options]
"""
from __future__ import annotations

import argparse
import sys

from lyricspike.util import setup_console


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spike.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check GPU / libraries (downloads nothing)")

    r = sub.add_parser("run", help="separate, transcribe, score and align one song")
    r.add_argument("song", help="audio file (mp3, flac, wav, m4a, ogg, ...)")
    r.add_argument("--out", help="output folder (default: out/<song name>)")
    r.add_argument("--lyrics", help="text file with the real lyrics: aligned directly (and used as the scoring reference)")
    r.add_argument("--reference", help="real lyrics used ONLY to score the transcripts (alignment then uses the ASR draft)")
    r.add_argument("--lang", default="en,ja", help="'en', 'ja', 'en,ja' (pick per region) or 'auto' (default: en,ja)")
    r.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    r.add_argument("--force", action="store_true", help="ignore cached stems")

    g = r.add_argument_group("separation")
    g.add_argument("--no-separate", action="store_true", help="skip vocal isolation")
    g.add_argument("--sep-model", action="append", help="separator checkpoint filename; repeat to ensemble "
                   "(default: vocals_mel_band_roformer.ckpt)")
    g.add_argument("--ensemble", action="store_true", help="also run a BS-Roformer and average the vocal stems (slower)")
    g.add_argument("--karaoke-model", nargs="?", const="mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
                   help="also split the vocals into lead and backing (adds 'lead'/'backing' variants)")
    g.add_argument("--sep-precision", default="fp16", choices=["fp16", "autocast", "fp32"])
    g.add_argument("--sep-segment", type=int, help="smaller value (e.g. 128) lowers separation memory use")
    g.add_argument("--sep-overlap", type=int, help="chunk overlap; larger = slower, slightly better")
    g.add_argument("--model-dir", help="where separator checkpoints are stored")

    g = r.add_argument_group("transcription")
    g.add_argument("--no-asr", action="store_true", help="skip transcription (e.g. when testing sync with --lyrics)")
    g.add_argument("--asr-model", default="large-v2", help="Whisper size/name (default large-v2; try medium if memory is tight)")
    g.add_argument("--asr-compute", help="CTranslate2 compute type (default int8_float16 on GPU, int8 on CPU)")
    g.add_argument("--beam", type=int, default=3)
    g.add_argument("--prompt", help="text that primes spelling/style, e.g. artist names and slang")
    g.add_argument("--variants", help="comma list from: mix_raw, mix_seg, vocals, lead, backing "
                   "(default: mix_raw plus every stem available)")
    g.add_argument("--pitch-shifts", default="0", help="comma list of semitone shifts to try per region, keeping "
                   "the most confident, e.g. '0,-3' (lowers high or pitched-up voices; slower; default: 0)")
    g.add_argument("--whisper-guards", action="store_true", help="re-enable Whisper's own 'not speech / unsure -> skip "
                   "the audio' checks on sung regions (off by default: they silently drop singing)")
    g.add_argument("--dump-regions", action="store_true", help="save the exact audio Whisper hears for each sung "
                   "region (asr/regions_<variant>/) so you can listen to it")

    g = r.add_argument_group("alignment")
    g.add_argument("--no-align", action="store_true")
    g.add_argument("--align-from", help="which transcript variant to align when no --lyrics is given (default: best)")
    g.add_argument("--align-on", default="vocals", choices=["vocals", "lead", "mix"])
    g.add_argument("--aligner-model", default="MahmoudAshraf/mms-300m-1130-forced-aligner")
    return p


def main(argv=None) -> int:
    setup_console()
    args = build_parser().parse_args(argv)
    if args.cmd == "doctor":
        from lyricspike.doctor import run_doctor

        return run_doctor()
    from lyricspike.pipeline import run

    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
