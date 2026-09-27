"""Pipeline ASR -> alineación -> diarización, en secuencia y liberando cada modelo antes del siguiente.

No sabe nada del CLI: recibe Settings y un Reporter para el progreso.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from dataclasses import dataclass, field

from .config import Settings


class DiarizationError(RuntimeError):
    """Se pidió diarización y no se pudo obtener hablantes para toda la transcripción."""


@dataclass
class Reporter:
    """Imprime el avance por etapa con sus tiempos."""

    total_stages: int
    timings: dict[str, float] = field(default_factory=dict)
    _index: int = 0

    def stage(self, key: str, label: str):
        return _Stage(self, key, label)

    def progress(self, pct: float) -> None:
        print(f"\r      {pct:5.1f}%", end="", file=sys.stderr, flush=True)


class _Stage:
    def __init__(self, reporter: Reporter, key: str, label: str):
        self.reporter, self.key, self.label = reporter, key, label

    def __enter__(self):
        self.reporter._index += 1
        print(f"[{self.reporter._index}/{self.reporter.total_stages}] {self.label}...", file=sys.stderr, flush=True)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, *_):
        elapsed = time.perf_counter() - self.t0
        self.reporter.timings[self.key] = round(elapsed, 1)
        status = "listo" if exc_type is None else "FALLÓ"
        print(f"\r      {status} en {fmt_duration(elapsed)}      ", file=sys.stderr, flush=True)


def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m} min {s:02d} s" if m else f"{seconds:.1f} s"


def free_memory(device: str) -> None:
    gc.collect()
    if device == "cuda":
        import torch

        torch.cuda.empty_cache()


def run(audio_path: str, s: Settings, device: str, reporter: Reporter) -> dict:
    import whisperx

    compute_type = s.compute_type_for(device)

    with reporter.stage("audio", "Cargando audio (ffmpeg)"):
        audio = whisperx.load_audio(audio_path)
    duration = len(audio) / 16000
    print(f"      duración del audio: {fmt_duration(duration)}", file=sys.stderr)

    with reporter.stage("asr", f"Transcribiendo ({s.model}, {device}, {compute_type}, batch {s.batch_size})"):
        asr_options = {"initial_prompt": s.initial_prompt} if s.initial_prompt else None
        model = whisperx.load_model(
            s.model, device, compute_type=compute_type, language=s.language, asr_options=asr_options
        )
        result = model.transcribe(
            audio, batch_size=s.batch_size, language=s.language, progress_callback=reporter.progress
        )
        del model
        free_memory(device)
    language = result["language"]

    with reporter.stage("alineacion", f"Alineando palabras (idioma {language})"):
        align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, device, progress_callback=reporter.progress
        )
        del align_model
        free_memory(device)

    if s.diarize:
        from whisperx.diarize import DiarizationPipeline

        with reporter.stage("diarizacion", "Identificando hablantes (pyannote)"):
            pipeline = DiarizationPipeline(
                model_name=s.diarization_model, token=os.environ.get("HF_TOKEN"), device=device
            )
            diarize_df = pipeline(
                audio,
                num_speakers=s.num_speakers,
                min_speakers=s.min_speakers,
                max_speakers=s.max_speakers,
                progress_callback=reporter.progress,
            )
            del pipeline
            free_memory(device)
            if diarize_df is None or len(diarize_df) == 0:
                raise DiarizationError("pyannote no detectó ningún turno de habla.")
            # fill_nearest: un segmento que cae en un hueco entre turnos de pyannote recibe el hablante más cercano.
            result = whisperx.assign_word_speakers(diarize_df, result, fill_nearest=True)
            missing = [seg for seg in result["segments"] if "speaker" not in seg]
            if missing:
                raise DiarizationError(
                    f"{len(missing)} segmentos quedaron sin hablante (primero en {missing[0]['start']:.1f} s)."
                )

    result["language"] = language
    result["duration"] = duration
    result["device"] = device
    result["compute_type"] = compute_type
    return result
