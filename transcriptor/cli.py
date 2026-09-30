"""CLI: python -m transcriptor <audio> [opciones]."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import FORMATS, apply_overrides, load_settings

EXIT_SETUP, EXIT_DIARIZATION = 2, 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m transcriptor",
        description="Transcribe reuniones en local con WhisperX. Los defaults están en config.toml.",
    )
    p.add_argument("audio", type=Path, help="archivo de audio o video (cualquier formato que lea ffmpeg)")
    p.add_argument("--config", type=Path, help="archivo de configuración alternativo (default: config.toml)")

    g = p.add_argument_group("diarización")
    g.add_argument("--diarize", action=argparse.BooleanOptionalAction, default=None,
                   help="identificar hablantes (--no-diarize para desactivar)")
    g.add_argument("--speakers", type=int, dest="num_speakers", metavar="N", help="número exacto de hablantes")
    g.add_argument("--min-speakers", type=int, metavar="N")
    g.add_argument("--max-speakers", type=int, metavar="N")

    g = p.add_argument_group("modelo")
    g.add_argument("--language", help='código de idioma (es, en...) o "auto"')
    g.add_argument("--model", help="modelo Whisper (large-v3, medium, ...)")
    g.add_argument("--device", choices=["auto", "cuda", "cpu"])
    g.add_argument("--compute-type", help='int8, int8_float16, float32... o "auto" (default: según la GPU)')
    g.add_argument("--batch-size", help='entero o "auto" (default: el mayor que cabe en la VRAM libre)')
    g.add_argument("--prompt", dest="initial_prompt", metavar="TEXTO",
                   help="texto de contexto con nombres y siglas (ver README: puede perder o inventar texto)")

    g = p.add_argument_group("salida")
    g.add_argument("--output-dir", type=Path, help="carpeta base; cada audio va a <output-dir>/<nombre>/")
    g.add_argument("--formats", help=f"lista separada por comas de: {','.join(FORMATS)} (el .json se guarda siempre)")
    g.add_argument("--timestamps", action=argparse.BooleanOptionalAction, default=None,
                   help="[hh:mm:ss] al inicio de cada turno del .txt")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_speakers is not None and (args.min_speakers is not None or args.max_speakers is not None):
        parser.error("usa --speakers (exacto) o --min/--max-speakers (rango), no ambos")
    overrides = vars(args).copy()
    if args.formats:
        overrides["formats"] = [f.strip() for f in args.formats.split(",") if f.strip()]
    try:
        settings = apply_overrides(load_settings(args.config), overrides)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error de configuración: {exc}", file=sys.stderr)
        return EXIT_SETUP
    if not args.audio.is_file():
        print(f"No existe el archivo: {args.audio}", file=sys.stderr)
        return EXIT_SETUP

    print("Preparando entorno...", file=sys.stderr, flush=True)
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))

    from . import runtime
    from .pipeline import DiarizationError, Reporter, fmt_duration, run
    from .writers import write_all

    runtime.setup()
    try:
        runtime.check_ffmpeg()
        device, warning = runtime.resolve_device(settings.device)
        if warning:
            print(f"AVISO: {warning}", file=sys.stderr)
        # Desde acá settings tiene valores concretos: "auto" se resuelve según la máquina.
        settings.compute_type = runtime.resolve_compute_type(device, settings.compute_type_for(device))
        settings.batch_size, warning = runtime.resolve_batch_size(device, settings.batch_size)
        if warning:
            print(f"AVISO: {warning}", file=sys.stderr)
        vram_free = runtime.free_vram_mb() if device == "cuda" else None
        if vram_free is not None:
            print(f"GPU: {vram_free} MiB libres -> {settings.compute_type}, batch {settings.batch_size}",
                  file=sys.stderr)
        if settings.diarize:
            import os

            runtime.check_diarization_access(settings.diarization_model, os.environ.get("HF_TOKEN"))
    except runtime.SetupError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_SETUP

    reporter = Reporter(total_stages=4 if settings.diarize else 3)
    t0 = time.perf_counter()
    try:
        result = run(str(args.audio), settings, device, reporter)
    except DiarizationError as exc:
        print(f"\nError: la diarización falló: {exc}\nNo se generaron salidas. "
              "Usa --no-diarize si quieres la transcripción sin hablantes.", file=sys.stderr)
        return EXIT_DIARIZATION
    total = time.perf_counter() - t0

    meta = {
        "audio": args.audio.name,
        "model": settings.model,
        "language": result["language"],
        "device": result["device"],
        "compute_type": result["compute_type"],
        "batch_size": result["batch_size"],
        "vram_free_mb": vram_free,
        "initial_prompt": settings.initial_prompt,
        "diarization": settings.diarization_model if settings.diarize else None,
        "num_speakers": settings.num_speakers,
        "min_speakers": settings.min_speakers,
        "max_speakers": settings.max_speakers,
        "duration_s": round(result["duration"], 1),
        "timings_s": reporter.timings | {"total": round(total, 1)},
    }
    paths = write_all(result, settings.output_dir, args.audio.stem, settings.formats, meta, settings.timestamps)

    speakers = sorted({seg["speaker"] for seg in result["segments"] if seg.get("speaker")})
    print(f"\nListo en {fmt_duration(total)} (audio de {fmt_duration(result['duration'])}).", file=sys.stderr)
    if settings.diarize:
        print(f"Hablantes: {', '.join(speakers)}", file=sys.stderr)
    for path in paths:
        print(f"  {path}", file=sys.stderr)
    return 0
