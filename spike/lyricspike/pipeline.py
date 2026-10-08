"""Orchestrates the spike: decode -> separate -> transcribe variants -> score -> align -> export."""
from __future__ import annotations

import json
import shutil
import time
import traceback
from pathlib import Path

from . import align, asr, audio, export, score, separate
from .util import GpuMonitor, Stages, cuda_available, free_gpu, log

DEFAULT_MODEL_DIR = Path.home() / ".cache" / "lyricspike" / "separator-models"


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if cuda_available() else "cpu"


def gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        return "unknown"


def _read(path: str | None) -> str | None:
    return Path(path).expanduser().read_text(encoding="utf-8-sig") if path else None


def _fmt_ts(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def run(args) -> Path:
    song = Path(args.song).expanduser().resolve()
    if not song.is_file():
        raise SystemExit(f"File not found: {song}")
    out = Path(args.out).expanduser() if args.out else Path("out") / song.stem
    work, stems_dir, asr_dir, align_dir = out / "work", out / "stems", out / "asr", out / "align"
    for d in (work, stems_dir, asr_dir, align_dir):
        d.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    monitor = GpuMonitor()
    monitor.start()
    stages = Stages(monitor)
    errors: list[str] = []
    ctx: dict = {"song": song.name, "device": device, "gpu": gpu_name(), "gpu_total": monitor.total_mb,
                 "lang": args.lang, "precision": args.sep_precision, "errors": errors, "files": [],
                 "sep_models": [], "asr": {}, "dropped": {}}

    def fail(stage: str, exc: Exception) -> None:
        traceback.print_exc()
        msg = f"{stage}: {type(exc).__name__}: {exc}"
        errors.append(msg)
        if "out of memory" in str(exc).lower():
            errors.append("  (GPU ran out of memory. Close other GPU apps, then try: --asr-model medium, --beam 1, "
                          "--sep-segment 128, or run stages separately with --no-asr / --no-separate.)")
        log(f"!! {msg}")

    lyrics_text = _read(args.lyrics)
    reference_text = _read(args.reference) or lyrics_text
    ctx["has_reference"] = bool(reference_text)

    # ------------------------------------------------------------------ decode
    with stages.stage("decode + copy original"):
        stereo = audio.decode(song, 44100, True)
        input_wav = work / "input_44k.wav"
        audio.write_wav(input_wav, stereo, 44100)
        ctx["duration"] = len(stereo) / 44100
        del stereo
        original = out / ("original" + song.suffix)
        shutil.copyfile(song, original)
        mix16 = audio.decode(song, 16000, False)

    # ------------------------------------------------------------------ separation
    stems: dict[str, Path] = {}
    model_dir = Path(args.model_dir).expanduser() if args.model_dir else DEFAULT_MODEL_DIR
    if not args.no_separate:
        models = list(args.sep_model or [separate.DEFAULT_VOCAL_MODEL])
        if args.ensemble and separate.SECOND_VOCAL_MODEL not in models:
            models.append(separate.SECOND_VOCAL_MODEL)
        ctx["sep_models"] = models
        try:
            with stages.stage(f"separate vocals ({len(models)} model{'s' if len(models) > 1 else ''})"):
                stems["vocals"] = separate.separate_vocals(
                    input_wav, stems_dir, models, model_dir, args.sep_precision, args.force,
                    args.sep_segment, args.sep_overlap)
        except Exception as exc:
            fail("vocal separation", exc)
        if args.karaoke_model and "vocals" in stems:
            try:
                with stages.stage("lead/backing split"):
                    stems["lead"], stems["backing"] = separate.split_lead_backing(
                        stems["vocals"], stems_dir, args.karaoke_model, model_dir, args.sep_precision, args.force)
            except Exception as exc:
                fail("lead/backing split", exc)

    waves: dict[str, object] = {"mix": mix16}

    def wave_of(key: str):
        if key not in waves:
            waves[key] = audio.decode(stems[key], 16000, False)
        return waves[key]

    # ------------------------------------------------------------------ transcription
    transcripts: dict[str, list[asr.Seg]] = {}
    if not args.no_asr:
        ctx["asr_model"] = args.asr_model
        specs = {"mix_raw": ("mix", None), "mix_seg": ("mix", "vocals"), "vocals": ("vocals", "vocals"),
                 "lead": ("lead", "lead"), "backing": ("backing", "backing")}
        wanted = [v.strip() for v in args.variants.split(",")] if args.variants else \
            ["mix_raw"] + [k for k in ("vocals", "lead", "backing") if k in stems]
        wanted = [v for v in wanted if v in specs]
        try:
            with stages.stage("load Whisper"):
                model = asr.load_model(args.asr_model, device, args.asr_compute)
            policy, opts = asr.parse_lang(args.lang), asr.AsrOptions(beam=args.beam, prompt=args.prompt)
            region_cache: dict[str, list] = {}
            for name in wanted:
                audio_key, seg_key = specs[name]
                needed = {audio_key} | ({seg_key} if seg_key else set())
                if any(k != "mix" and k not in stems for k in needed):
                    log(f"skipping variant '{name}': the stem it needs was not produced")
                    continue
                t0 = time.perf_counter()
                try:
                    with stages.stage(f"transcribe [{name}]"):
                        if seg_key is None:
                            segs, regions = asr.transcribe_full(model, wave_of(audio_key), policy, opts), None
                        else:
                            if seg_key not in region_cache:
                                region_cache[seg_key] = audio.vocal_segments(wave_of(seg_key))
                            regions = region_cache[seg_key]
                            log(f"  {len(regions)} sung regions found in the {seg_key} stem")
                            segs = asr.transcribe_slices(model, wave_of(audio_key), regions, policy, opts)
                except Exception as exc:
                    fail(f"transcription [{name}]", exc)
                    continue
                kept, dropped = asr.filter_segments(segs)
                transcripts[name] = kept
                ctx["dropped"][name] = [(s.text, why) for s, why in dropped]
                text = "\n".join(f"[{_fmt_ts(s.start)}] {s.text}" for s in kept)
                (asr_dir / f"{name}.txt").write_text(text + "\n", encoding="utf-8")
                (asr_dir / f"{name}.json").write_text(
                    json.dumps([s.to_dict() for s in kept], ensure_ascii=False, indent=1), encoding="utf-8")
                metrics = score.evaluate(reference_text, " ".join(s.text for s in kept)) if reference_text else None
                ctx["asr"][name] = {"lines": len(kept), "seconds": time.perf_counter() - t0, "metrics": metrics,
                                    "regions": None if regions is None else len(regions)}
                ctx["files"] += [f"asr/{name}.txt", f"asr/{name}.json"]
            del model
            free_gpu()
        except Exception as exc:
            fail("transcription setup", exc)

    # ------------------------------------------------------------------ alignment
    lines = None
    if not args.no_align and (lyrics_text or transcripts):
        align_key = args.align_on if args.align_on in stems else ("vocals" if "vocals" in stems else None)
        notes: list[str] = []
        if align_key is None:
            notes.append("no isolated vocal stem available: aligning against the full mix (expect worse sync)")
        wave = wave_of(align_key) if align_key else mix16
        try:
            with stages.stage("load aligner"):
                aligner = align.load_aligner(args.aligner_model, device)
            notes.append(f"aligner: {aligner.description}; audio: {align_key or 'mix'}")
            with stages.stage("align lyrics to audio"):
                emissions = align.compute_emissions(wave, aligner.infer)
                if lyrics_text:
                    lines = align.build_lines(align.parse_lyrics(lyrics_text))
                    dropped_chars = align.align_lyrics(emissions, lines, aligner)
                    source = "pasted lyrics, aligned globally"
                else:
                    ref_name = _choose_draft(args.align_from, transcripts, ctx["asr"])
                    lines, failures = align.align_draft(emissions, transcripts[ref_name], aligner)
                    dropped_chars = 0
                    source = f"ASR draft from '{ref_name}' variant, aligned per region"
                    if failures:
                        notes.append(f"{failures} region(s) could not be aligned and were spread evenly")
                if dropped_chars:
                    notes.append(f"{dropped_chars} character(s) were not in the aligner vocabulary and were skipped")
            del aligner
            free_gpu()
            ctx["align_mode"], ctx["align_notes"], ctx["lines"] = source, notes, lines
            (align_dir / "lyrics.lrc").write_text(export.to_lrc(lines), encoding="utf-8")
            (align_dir / "lyrics.enhanced.lrc").write_text(export.to_enhanced_lrc(lines), encoding="utf-8")
            data = export.lines_to_dict(lines, source)
            (align_dir / "aligned.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            sources = [("original mix", original.name)] + [
                (f"{k} (isolated)", f"stems/{k}.wav") for k in ("vocals", "lead", "backing") if k in stems]
            export.write_viewer(out / "viewer.html", data, sources, song.stem)
            ctx["files"] += ["align/lyrics.lrc", "align/lyrics.enhanced.lrc", "align/aligned.json", "viewer.html"]
        except Exception as exc:
            fail("alignment", exc)

    # ------------------------------------------------------------------ report
    monitor.stop()
    ctx["stages"] = stages.rows
    export.write_report(out / "report.md", ctx)
    log(f"report: {out / 'report.md'}")
    if lines is not None:
        log(f"karaoke preview: open {out / 'viewer.html'} in a browser")
    return out / "report.md"


def _choose_draft(preferred: str | None, transcripts: dict, asr_ctx: dict) -> str:
    if preferred and preferred in transcripts:
        return preferred
    scored = [(row["metrics"]["pcer"], name) for name, row in asr_ctx.items()
              if name in transcripts and row.get("metrics") and row["metrics"].get("pcer") is not None]
    if scored:
        return min(scored)[1]
    for name in ("lead", "vocals", "mix_seg", "mix_raw"):
        if name in transcripts:
            return name
    return next(iter(transcripts))
