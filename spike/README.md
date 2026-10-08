# lyricspike: a feasibility spike

A small command-line test rig for one question: **can a fully local pipeline transcribe and time-sync the lyrics of
heavily produced songs (hyperpop, digicore, English + Japanese) well enough to build an app around?**

It is not the app. There is no GUI. It runs the real pipeline on one song at a time and tells you, with numbers
and by ear, where it works and where it does not.

```
song.mp3 ──► decode ──► vocal isolation (Mel-Band Roformer) ──► find sung regions ──► Whisper large-v2
                                │                                                       │
                                └──────────► CTC forced alignment (MMS) ◄───── lyrics (yours, or the Whisper draft)
                                                      │
                          report.md · lyrics.lrc · word-level LRC · viewer.html (karaoke preview)
```

It compares **raw mix vs isolated vocals** for transcription, scores each against real lyrics if you give them
(WER / CER / phonetic CER), and aligns lyrics to the audio so you can watch the words light up in time with the music.

## 1. One-time setup (Windows 10/11, NVIDIA GPU)

You need Python **3.12** from python.org (3.13 may lack wheels for some packages). Open PowerShell:

```powershell
cd path\to\Local-transcribee\spike

py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip

# PyTorch FIRST, the CUDA build. Get the exact command from https://pytorch.org/get-started/locally/
# Choose Stable / Windows / Pip / a CUDA *12.x* build. It looks like this:
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
python spike.py doctor
```

`doctor` downloads nothing. It checks that torch sees your GPU, that the Whisper engine (CTranslate2) can use CUDA,
that every library imports, and that ffmpeg is available (a bundled copy is used if you have none). Fix anything it
marks `FAIL` before going on. If it says torch cannot see a GPU, you installed the CPU wheel: reinstall from the CUDA index.

The first real run downloads roughly **5 GB** of models: Roformer 0.9 GB (stored under `%USERPROFILE%\.cache\lyricspike`), Whisper
large-v2 ~3 GB and the aligner ~1.2 GB (stored in the Hugging Face cache). `--ensemble` adds a 0.64 GB BS-Roformer and
`--karaoke-model` a 0.9 GB karaoke model. Stage timings from that first run include the downloads, so run a song twice for honest timings.

## 2. Run it

Put the song somewhere and, if you have them, the real lyrics in a UTF-8 text file (one sung line per line;
`[Chorus]`-style header lines are ignored; Japanese is fine).

**A. How good is transcription on my songs?** (needs the real lyrics only to score it)
```powershell
python spike.py run "C:\music\song.mp3" --reference "C:\music\song_lyrics.txt"
```
Runs Whisper on the raw mix **and** on the isolated vocals, scores both, then aligns the better draft.

**B. How good is the sync?** (skips Whisper: fastest, and the most reliable path)
```powershell
python spike.py run "C:\music\song.mp3" --lyrics "C:\music\song_lyrics.txt" --no-asr
```

**C. Try the stronger options** (stems are cached per model, so repeats are cheaper)
```powershell
python spike.py run song.mp3 --reference lyrics.txt --ensemble          # + BS-Roformer, averaged
python spike.py run song.mp3 --reference lyrics.txt --karaoke-model     # also lead / backing vocal variants
python spike.py run song.mp3 --reference lyrics.txt --asr-model large-v3
python spike.py run song.mp3 --lang ja --prompt "Japanese lyrics. Artist: ..."
```

Everything lands in `out\<song name>\`.

## 3. Reading the results

1. **`viewer.html`**, open in a browser. Play the song; the current word fills with colour as it is sung.
   Use the **Audio** menu to listen to the isolated vocals instead (much easier to judge sync).
   Drag **Sync offset**: if lyrics are consistently early/late by a fixed amount, that is a calibration issue,
   not misalignment. A dotted underline marks words the aligner was unsure about. Click any word to jump to it.
2. **`report.md`**: stage timings and peak GPU memory (does it fit in 4 GB?), the WER/CER table per variant, and
   the ten least-confident lines to check first.
3. `asr\*.txt`: each transcript variant with timestamps, so you can read raw vs isolated side by side.
4. `align\lyrics.lrc` and `lyrics.enhanced.lrc` (word-level) work in most lyric-capable players.

**WER** counts words (meaningless for Japanese). **CER** counts characters. **pCER** compares romanized text, so
`夜` vs `よる` is not an error; use it for Japanese. Lower is better. If the reference writes a repeated chorus once
but the song sings it three times, scores will look worse than the transcript really is.

## 4. What to send back

- `report.md` (and the console output if something failed)
- `align\aligned.json`
- a few sentences per song: which lines were wrong, whether sync felt early/late/drifting, and anything weird you heard

Ideal test set: one clean-vocal song, one very processed hyperpop/digicore track, one Japanese (or mixed) track.

## 5. Troubleshooting

| Symptom | Try |
|---|---|
| `out of memory` (any stage) | close other GPU apps (browsers, games); `--asr-model medium`; `--beam 1`; `--sep-segment 128`; run stages separately with `--no-asr` / `--no-separate` |
| `Could not load library cudnn...` / `cublas64_12.dll not found` | torch must be a CUDA **12.x** build and `doctor` should show CTranslate2 with 1 CUDA device |
| `FFmpeg` errors | the spike falls back to a bundled copy; or `winget install Gyan.FFmpeg` |
| Alignment looks scrambled with `--lyrics` | the lyrics probably contain lines not actually sung (or are missing some); the least-confident lines in `report.md` show where it went wrong |
| Very slow | expected on a 4 GB laptop GPU with `--ensemble` / `--karaoke-model`; compare the stage times in `report.md` |

## 6. Known limits of this spike (on purpose)

- Japanese romanization uses a dictionary tokenizer: unusual kanji readings and stylized lyrics can be wrong, so
  Japanese sync is the least reliable part. Treat it as beta.
- Digits are aligned as if spoken digit by digit ("808" as eight-zero-eight).
- Whole-song alignment of pasted lyrics can drift after a lyric that is missing or wrong. A production version
  would re-anchor periodically.
- The karaoke split (lead/backing) is experimental. The log prints which stem it treated as lead: verify by ear.
- Whisper large-v2 is the default because it is documented to behave better than v3 on singing; other ASR models
  (for example Qwen3-ASR) are not wired in yet and may not fit in 4 GB without quantization.
- No GUI, batching, cancel/resume, or editor. Those come after we know this core works.

## What has and has not been verified

Built and tested in a cloud sandbox with no GPU and no access to Hugging Face, so:

| Verified for real | Still unproven: this is what your run tells us |
|---|---|
| Decoding mp3 / flac / ogg / m4a / wav | Whisper large-v2 on your songs: accuracy **and** whether it fits in 4 GB |
| The real Mel-Band Roformer, BS-Roformer and karaoke models run through the spike (on CPU): downloads, output naming, ensemble averaging, lead/backing stem classification | The aligner model's real weights (its repo name is from memory; a vocabulary sanity check will fail loudly if it is wrong) |
| Separation quality on a speech-over-synthetic-music test: ~24 dB SDR, music leakage ~0 | GPU speed and peak memory of any stage on an RTX 500 Ada |
| Sung-region detection finds the phrases in the isolated stem but one giant region in the raw mix | Anything about real hyperpop/digicore: the test music was easy and synthetic |
| CTC alignment maths (matches an exhaustive brute-force search), window stitching, Japanese romanization | Japanese sync quality |
| Aligner plumbing with real `transformers` 5.x on a tiny stand-in model; Whisper call signature against real `faster-whisper` | A clean install on Windows |
| The karaoke viewer in a real browser (highlighting, seeking, offset, Japanese) | |
| Whole pipeline wiring, report and exports, with the Whisper/aligner models swapped for stand-ins | |

Bugs this verification already caught and fixed: the newest PyAV breaks faster-whisper's own audio loader (so the spike
decodes audio itself), `audio-separator` needs an undeclared `audioread` and an `ffmpeg` command (the spike bundles one),
and a lead/backing mix-up risk for karaoke models.

## Developing

```
pip install pytest
python -m pytest tests -q
```
The tests cover the maths (CTC alignment checked against brute force, window stitching), text handling, scoring,
decoding of mp3/flac/ogg/m4a, export, the karaoke viewer's data path, and the whole pipeline wiring with the heavy
models swapped for stand-ins. They do not exercise the real models; that is what `spike.py run` is for.
