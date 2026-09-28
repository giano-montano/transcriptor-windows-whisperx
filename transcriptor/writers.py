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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    """Escribe en <out_dir>/<stem>/<stem>.<fmt>. El .json se guarda siempre: es la base de transcriptor.render."""
    folder = out_dir / stem
    folder.mkdir(parents=True, exist_ok=True)
    json_path = folder / f"{stem}.json"
    write_json(result, json_path, meta)
    return [*render(result, folder, stem, [f for f in formats if f != "json"], timestamps), json_path]


def apply_speaker_names(segments: list[dict], names: dict[str, str]) -> list[dict]:
    """Cambia las etiquetas (SPEAKER_00...) por nombres. Dos etiquetas con el mismo nombre quedan unidas."""
    if not names:
        return segments
    return [seg | {"speaker": names.get(seg["speaker"], seg["speaker"])} if seg.get("speaker") else seg
            for seg in segments]


def render(result: dict, folder: Path, stem: str, formats: list[str], timestamps: bool) -> list[Path]:
    """Escribe las salidas legibles (txt, srt) a partir de los segmentos, con los nombres de meta.speaker_names."""
    names = result.get("meta", {}).get("speaker_names", {})
    result = {"segments": apply_speaker_names(result["segments"], names)}
    written = []
    for fmt in formats:
        path = folder / f"{stem}.{fmt}"
        if fmt == "txt":
            write_txt(result, path, timestamps)
        elif fmt == "srt":
            write_srt(result, path)
        else:
            raise ValueError(f"Formato no soportado: {fmt}")
        written.append(path)
    return written
