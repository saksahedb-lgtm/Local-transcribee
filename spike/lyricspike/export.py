"""LRC / JSON / report / HTML-viewer writers."""
from __future__ import annotations

import json
from pathlib import Path

from .align import Line

LOW_SCORE = -4.0  # mean log-prob below this is flagged as shaky in the viewer and report


def lrc_ts(t: float, open_: str = "[", close: str = "]") -> str:
    cs = max(0, int(round(t * 100)))
    return f"{open_}{cs // 6000:02d}:{(cs % 6000) / 100:05.2f}{close}"


def to_lrc(lines: list[Line]) -> str:
    return "\n".join(f"{lrc_ts(ln.start)}{ln.text}" for ln in lines if ln.start is not None) + "\n"


def to_enhanced_lrc(lines: list[Line]) -> str:
    out = []
    for ln in lines:
        if ln.start is None:
            continue
        words = " ".join(f"{lrc_ts(u.start, '<', '>')}{u.text}" for u in ln.units if u.start is not None)
        out.append(f"{lrc_ts(ln.start)} {words}")
    return "\n".join(out) + "\n"


def lines_to_dict(lines: list[Line], source: str) -> dict:
    return {
        "source": source,
        "lines": [
            {
                "text": ln.text, "start": ln.start, "end": ln.end, "score": ln.score,
                "words": [{"text": u.text, "start": u.start, "end": u.end, "score": u.score} for u in ln.units],
            }
            for ln in lines
        ],
    }


def write_viewer(path: Path, data: dict, audio_sources: list[tuple[str, str]], title: str) -> None:
    """Self-contained karaoke preview. `audio_sources` are (label, path relative to the viewer file)."""
    template = (Path(__file__).with_name("viewer.html")).read_text(encoding="utf-8")
    payload = json.dumps({"title": title, "audio": audio_sources, "data": data, "low": LOW_SCORE},
                         ensure_ascii=False).replace("</", "<\\/")
    path.write_text(template.replace("/*__PAYLOAD__*/null", payload), encoding="utf-8")


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def write_report(path: Path, ctx: dict) -> None:
    L: list[str] = []
    L.append(f"# lyricspike report: {ctx['song']}\n")
    L.append(f"- duration: {ctx.get('duration', 0):.1f}s")
    L.append(f"- device: {ctx.get('device')}; GPU: {ctx.get('gpu', 'n/a')}")
    L.append(f"- separation: {', '.join(ctx.get('sep_models', [])) or 'skipped'} (precision {ctx.get('precision')})")
    L.append(f"- ASR: {ctx.get('asr_model', 'skipped')}; language policy: {ctx.get('lang')}")
    L.append(f"- alignment mode: {ctx.get('align_mode', 'skipped')}\n")

    L.append("## Stage timings\n")
    L.append("| stage | time | peak GPU memory in use |")
    L.append("|---|---|---|")
    gpu_total = ctx.get("gpu_total")
    for name, dt, peak in ctx.get("stages", []):
        mem = "n/a" if peak is None else "%d MB / %s MB" % (peak, gpu_total)
        L.append(f"| {name} | {dt:.1f}s | {mem} |")
    L.append("\n(GPU memory is whole-GPU usage sampled once per second, so it includes other apps and can miss brief peaks.)\n")

    if ctx.get("asr"):
        L.append("## Transcription variants\n")
        has_ref = ctx.get("has_reference")
        L.append("| variant | regions | lines | time | " + ("WER | CER | pCER |" if has_ref else ""))
        L.append("|---|---|---|---|" + ("---|---|---|" if has_ref else ""))
        for name, row in ctx["asr"].items():
            m = row.get("metrics") or {}
            tail = f" {pct(m.get('wer'))} | {pct(m.get('cer'))} | {pct(m.get('pcer'))} |" if has_ref else ""
            regions = '-' if row.get('regions') is None else row['regions']
            L.append(f"| {name} | {regions} | {row['lines']} | {row['seconds']:.1f}s |{tail}")
        if has_ref:
            L.append("\nWER counts words (not meaningful for Japanese). CER counts characters. pCER compares "
                     "romanized text, so kanji-vs-kana spelling differences do not count as errors. "
                     "Lower is better. Repeated choruses abbreviated in the reference will inflate the numbers.")
        dropped = ctx.get("dropped", {})
        if dropped:
            L.append("\nSegments removed by the hallucination filter:")
            for name, items in dropped.items():
                for text, reason in items[:8]:
                    L.append(f"- {name}: \"{text}\" ({reason})")
        L.append("")

    if ctx.get("lines") is not None:
        lines: list[Line] = ctx["lines"]
        scored = [ln for ln in lines if ln.score is not None]
        L.append("## Alignment\n")
        L.append(f"- lines: {len(lines)}; words/units: {sum(len(ln.units) for ln in lines)}")
        if scored:
            L.append(f"- mean confidence (log-prob, closer to 0 is better): "
                     f"{sum(ln.score for ln in scored) / len(scored):.2f}")
            L.append(f"- lines flagged shaky (< {LOW_SCORE}): {sum(1 for ln in scored if ln.score < LOW_SCORE)}")
        for note in ctx.get("align_notes", []):
            L.append(f"- {note}")
        L.append("\nLeast confident lines (check these by ear first):\n")
        for ln in sorted(scored, key=lambda x: x.score)[:10]:
            L.append(f"- {lrc_ts(ln.start)} score {ln.score:.2f}: {ln.text}")
        L.append("")

    if ctx.get("errors"):
        L.append("## Problems\n")
        L.extend(f"- {e}" for e in ctx["errors"])
        L.append("")

    L.append("## Files\n")
    for f in ctx.get("files", []):
        L.append(f"- `{f}`")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
