"""Escritura de salidas: .txt legible de reunión, .srt y .json con timestamps por palabra."""

from __future__ import annotations

import json
from pathlib import Path


def hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def meeting_turns(segments: list[dict]) -> list[tuple[float, str | None, str]]:
    """Fusiona segmentos consecutivos del mismo hablante en un solo turno."""
    turns: list[tuple[float, str | None, str]] = []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        speaker = seg.get("speaker")
        if turns and speaker is not None and turns[-1][1] == speaker:
            start, _, prev = turns[-1]
            turns[-1] = (start, speaker, f"{prev} {text}")
        else:
            turns.append((seg["start"], speaker, text))
    return turns


def write_txt(result: dict, path: Path, timestamps: bool) -> None:
    lines = []
    for start, speaker, text in meeting_turns(result["segments"]):
        prefix = f"[{hms(start)}] " if timestamps else ""
        if speaker:
            prefix += f"{speaker}: "
        lines.append(prefix + text)
    path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def write_srt(result: dict, path: Path) -> None:
    blocks = []
    for i, seg in enumerate(result["segments"], 1):
        text = seg["text"].strip()
        if seg.get("speaker"):
            text = f"[{seg['speaker']}] {text}"
        blocks.append(f"{i}\n{srt_time(seg['start'])} --> {srt_time(seg['end'])}\n{text}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")


def write_json(result: dict, path: Path, meta: dict) -> None:
    payload = {"meta": meta, "segments": result["segments"], "word_segments": result.get("word_segments", [])}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def write_all(
    result: dict, out_dir: Path, stem: str, formats: list[str], meta: dict, timestamps: bool
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        if fmt == "txt":
            write_txt(result, path, timestamps)
        elif fmt == "srt":
            write_srt(result, path)
        elif fmt == "json":
            write_json(result, path, meta)
        written.append(path)
    return written
