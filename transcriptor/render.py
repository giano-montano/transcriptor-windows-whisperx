"""Regenera .txt/.srt desde el .json de una transcripción, sin volver a transcribir.

python -m transcriptor.render outputs\\reunion [--timestamps] [--formats txt,srt]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings, load_settings
from .writers import render

RENDER_FORMATS = ("txt", "srt")


def find_json(target: Path) -> Path:
    """Acepta el .json o la carpeta de la transcripción (outputs\\<nombre>)."""
    if target.is_dir():
        candidate = target / f"{target.name}.json"
        if candidate.is_file():
            return candidate
        found = sorted(target.glob("*.json"))
        if len(found) == 1:
            return found[0]
        raise FileNotFoundError(f"No se encontró un único .json en {target}")
    if target.suffix.lower() == ".json" and target.is_file():
        return target
    raise FileNotFoundError(f"No existe la carpeta o el .json: {target}")


def load_transcript(json_path: Path) -> dict:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if "segments" not in data:
        raise ValueError(f"{json_path} no es un .json de transcriptor (falta 'segments').")
    return data


def add_output_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("target", type=Path, help="carpeta de la transcripción (outputs\\<nombre>) o su .json")
    p.add_argument("--config", type=Path, help="archivo de configuración alternativo (default: config.toml)")
    p.add_argument("--formats", help=f"lista separada por comas de: {','.join(RENDER_FORMATS)} (default: config.toml)")
    p.add_argument("--timestamps", action=argparse.BooleanOptionalAction, default=None,
                   help="[hh:mm:ss] al inicio de cada turno del .txt")


def output_options(args: argparse.Namespace, settings: Settings,
                   parser: argparse.ArgumentParser) -> tuple[list[str], bool]:
    if args.formats:
        formats = [f.strip() for f in args.formats.split(",") if f.strip()]
        unknown = [f for f in formats if f not in RENDER_FORMATS]
        if unknown or not formats:
            parser.error(f"formatos no válidos: {unknown or formats}. Opciones: {', '.join(RENDER_FORMATS)}")
    else:
        formats = [f for f in settings.formats if f in RENDER_FORMATS] or ["txt"]
    timestamps = settings.timestamps if args.timestamps is None else args.timestamps
    return formats, timestamps


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m transcriptor.render",
        description="Regenera .txt/.srt desde el .json de una transcripción ya hecha, sin volver a transcribir. "
        "Los archivos se escriben junto al .json y reemplazan a los anteriores.",
    )
    add_output_args(p)
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

    for path in render(data, json_path.parent, json_path.stem, formats, timestamps):
        print(f"  {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
