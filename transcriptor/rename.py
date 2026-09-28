"""Pone nombres a los hablantes de una transcripción ya hecha, sin volver a transcribir.

python -m transcriptor.rename outputs\\reunion                    (pregunta el nombre de cada hablante)
python -m transcriptor.rename outputs\\reunion --names "SPEAKER_00=Ana,SPEAKER_01=Luis,SPEAKER_02=Luis"

Los nombres se guardan en el .json (meta.speaker_names) sin tocar las etiquetas originales, así que se pueden
corregir o deshacer después. Dar el mismo nombre a dos etiquetas las une.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_settings
from .render import add_output_args, find_json, load_transcript, output_options
from .writers import hms, render

SAMPLES = 3
SAMPLE_CHARS = 160


def speaker_labels(segments: list[dict]) -> list[str]:
    return sorted({seg["speaker"] for seg in segments if seg.get("speaker")})


def parse_names(spec: str, labels: list[str], names: dict[str, str]) -> dict[str, str]:
    """Aplica "CLAVE=Nombre,..." sobre names. CLAVE es una etiqueta original o un nombre ya puesto; Nombre vacío
    devuelve la etiqueta original."""
    names = dict(names)
    for item in filter(None, (part.strip() for part in spec.split(","))):
        key, sep, value = (s.strip() for s in item.partition("="))
        if not sep or not key:
            raise ValueError(f'"{item}" no tiene la forma ETIQUETA=Nombre')
        targets = [key] if key in labels else [label for label in labels if names.get(label) == key]
        if not targets:
            raise ValueError(f'"{key}" no es un hablante de esta transcripción. Hablantes: {", ".join(labels)}')
        for label in targets:
            names[label] = value
    return {label: name for label, name in names.items() if name and name != label}


def describe(segments: list[dict], label: str) -> str:
    own = [seg for seg in segments if seg.get("speaker") == label and seg["text"].strip()]
    seconds = sum(seg["end"] - seg["start"] for seg in own)
    longest = sorted(own, key=lambda seg: len(seg["text"].split()), reverse=True)[:SAMPLES]
    lines = [f"{len(own)} segmentos, {round(seconds / 60)} min de habla"]
    for seg in sorted(longest, key=lambda seg: seg["start"]):
        text = seg["text"].strip()
        if len(text) > SAMPLE_CHARS:
            text = text[:SAMPLE_CHARS].rstrip() + "..."
        lines.append(f"  [{hms(seg['start'])}] {text}")
    return "\n".join(lines)


def ask_names(segments: list[dict], labels: list[str], names: dict[str, str]) -> dict[str, str]:
    print("Enter deja el nombre actual; '-' vuelve a la etiqueta original. "
          "Mismo nombre en dos etiquetas = misma persona.", file=sys.stderr)
    names = dict(names)
    for label in labels:
        current = names.get(label, label)
        print(f"\n{label}" + (f" (ahora: {current})" if current != label else ""), file=sys.stderr)
        print(describe(segments, label), file=sys.stderr)
        answer = input(f"Nombre [{current}]: ").strip()
        if answer == "-":
            names.pop(label, None)
        elif answer:
            names[label] = answer
    return {label: name for label, name in names.items() if name != label}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m transcriptor.rename",
        description="Pone nombres a los hablantes de una transcripción ya hecha y regenera el .txt/.srt, sin volver "
        "a transcribir. Sin --names, muestra frases de cada hablante y pregunta su nombre.",
    )
    add_output_args(p)
    p.add_argument("--names", metavar='"ETIQUETA=Nombre,..."',
                   help='p. ej. "SPEAKER_00=Ana,SPEAKER_01=Luis,SPEAKER_02=Luis" (mismo nombre = misma persona)')
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
        json_path = find_json(args.target)
        data = load_transcript(json_path)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    formats, timestamps = output_options(args, settings, parser)

    segments = data["segments"]
    labels = speaker_labels(segments)
    if not labels:
        print("Error: la transcripción no tiene hablantes (se hizo con --no-diarize).", file=sys.stderr)
        return 2
    meta = data.setdefault("meta", {})
    try:
        if args.names:
            names = parse_names(args.names, labels, meta.get("speaker_names", {}))
        else:
            names = ask_names(segments, labels, meta.get("speaker_names", {}))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, EOFError):
        print("\nCancelado: no se cambió nada.", file=sys.stderr)
        return 1

    meta["speaker_names"] = names
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\nHablantes:", file=sys.stderr)
    for label in labels:
        print(f"  {label} -> {names.get(label, label)}", file=sys.stderr)
    for path in [*render(data, json_path.parent, json_path.stem, formats, timestamps), json_path]:
        print(f"  {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
