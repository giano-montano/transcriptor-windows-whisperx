"""Spike de Fase 0: ASR -> alineación -> diarización en secuencia, con tiempos y VRAM pico.

No es la herramienta final. Uso:
    uv run python scripts/spike.py samples/archivo.mp3 [--no-diarize]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import threading
import time
from pathlib import Path

# torch primero: carga cublas64_12.dll, que ctranslate2 no trae en Windows.
import torch
import whisperx
from dotenv import load_dotenv
import huggingface_hub.file_download as hf_fd
from whisperx.diarize import DiarizationPipeline


class VramSampler(threading.Thread):
    """Muestrea la VRAM total usada (nvidia-smi) para captar también a CTranslate2."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.peak = 0
        self.stop_event = threading.Event()

    def read(self) -> int:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        return int(out.stdout.split()[0])

    def run(self) -> None:
        while not self.stop_event.is_set():
            self.peak = max(self.peak, self.read())
            time.sleep(0.5)

    def stop(self) -> int:
        self.stop_event.set()
        self.join()
        return self.peak


def free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def stage(name, fn, results):
    sampler = VramSampler()
    baseline = sampler.read()
    sampler.start()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = fn()
    elapsed = time.perf_counter() - t0
    peak = sampler.stop()
    results[name] = {
        "segundos": round(elapsed, 1),
        "vram_total_pico_MiB": peak,
        "vram_base_MiB": baseline,
        "torch_pico_MiB": round(torch.cuda.max_memory_allocated() / 2**20),
    }
    print(f"[{name}] {elapsed:.1f}s  VRAM pico {peak} MiB (base {baseline})", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--no-diarize", action="store_true")
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()

    load_dotenv()
    # Race en huggingface_hub 0.36 (ver docs/TROUBLESHOOTING.md): se detecta una vez y se fija el resultado.
    _symlinks = hf_fd.are_symlinks_supported()
    hf_fd.are_symlinks_supported = lambda cache_dir=None: _symlinks
    device, compute_type = "cuda", "int8"
    results: dict = {}
    out_dir = Path("outputs/spike")
    out_dir.mkdir(parents=True, exist_ok=True)

    audio = stage("carga_audio", lambda: whisperx.load_audio(args.audio), results)
    print(f"Duración: {len(audio) / 16000:.1f}s")

    def asr():
        model = whisperx.load_model("large-v3", device, compute_type=compute_type, language="es")
        res = model.transcribe(audio, batch_size=args.batch_size, language="es")
        del model; free()
        return res

    result = stage("asr", asr, results)

    def align():
        model_a, meta = whisperx.load_align_model(language_code="es", device=device)
        res = whisperx.align(result["segments"], model_a, meta, audio, device)
        del model_a; free()
        return res

    result = stage("alineacion", align, results)

    if not args.no_diarize:
        def diarize():
            pipe = DiarizationPipeline(token=os.environ["HF_TOKEN"], device=device)
            df = pipe(audio)
            del pipe; free()
            return df

        diarize_df = stage("diarizacion", diarize, results)
        result = whisperx.assign_word_speakers(diarize_df, result)
        print("Hablantes:", sorted(diarize_df["speaker"].unique()))

    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    with open(out_dir / "result.txt", "w", encoding="utf-8") as f:
        for s in result["segments"]:
            f.write(f"[{s['start']:7.1f}] {s.get('speaker', '?')}: {s['text'].strip()}\n")
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
