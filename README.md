# Local-transcribee

A local (offline) tool for transcribing and time-syncing the lyrics of heavily produced songs
(hyperpop, digicore, English + Japanese): isolate the vocals with a Roformer model, transcribe them with a
Whisper model, and align the words to the audio so lyrics highlight in time with the music.

**Status: feasibility spike.** Before building the desktop app, [`spike/`](spike/README.md) holds a command-line rig that
runs the real pipeline on your songs and reports how well it works (accuracy numbers, GPU memory use, and a karaoke
preview for judging sync by ear). Start there.
