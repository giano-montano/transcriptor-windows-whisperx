"""Diagnóstico del entorno para transcriptor-windows-whisperx.

Solo usa la biblioteca estándar: debe funcionar antes de crear el entorno virtual.
Si torch está instalado, además reporta su build CUDA y prueba una operación real en GPU.

Uso:  python scripts/diagnose.py
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys


def run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def section(title: str) -> None:
    print(f"\n== {title}")


def report_system() -> None:
    section("Sistema")
    print(f"SO:        {platform.system()} {platform.release()} (build {platform.version()})")
    print(f"Máquina:   {platform.machine()}")
    print(f"Python:    {sys.version.split()[0]} ({sys.executable})")
    ok = (3, 10) <= sys.version_info[:2] < (3, 14)
    print(f"           {'OK' if ok else 'NO SOPORTADO'}: whisperx 3.8.x exige Python >=3.10,<3.14")


def report_gpu() -> None:
    section("GPU NVIDIA (nvidia-smi)")
    if not shutil.which("nvidia-smi"):
        print("nvidia-smi no encontrado: sin GPU NVIDIA o sin driver. Se usará CPU.")
        return
    query = "name,compute_cap,driver_version,memory.total,memory.used"
    out = run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"])
    if out is None:
        print("nvidia-smi falló al consultar la GPU.")
        return
    for line in out.splitlines():
        name, cc, driver, total, used = (x.strip() for x in line.split(","))
        print(f"GPU:       {name}")
        print(f"CC:        {cc}")
        print(f"Driver:    {driver}")
        print(f"VRAM:      {total} (en uso: {used})")
    header = run(["nvidia-smi"]) or ""
    for line in header.splitlines():
        if "CUDA Version" in line:
            print(f"CUDA máx. soportada por el driver: {line.split('CUDA Version:')[1].strip(' |')}")


def report_ffmpeg() -> None:
    section("ffmpeg")
    path = shutil.which("ffmpeg")
    if not path:
        print("ffmpeg NO está en PATH (whisperx lo invoca por línea de comandos).")
        return
    version = (run(["ffmpeg", "-version"]) or "").splitlines()
    print(f"Ruta:      {path}")
    print(f"Versión:   {version[0] if version else '?'}")


def report_torch() -> None:
    section("PyTorch")
    try:
        import torch
    except ImportError:
        print("torch no instalado en este intérprete (normal antes de crear el entorno).")
        return
    print(f"torch:     {torch.__version__}  (CUDA de la build: {torch.version.cuda})")
    print(f"cuda.is_available(): {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return
    cap = torch.cuda.get_device_capability(0)
    arch = f"sm_{cap[0]}{cap[1]}"
    archs = torch.cuda.get_arch_list()
    print(f"Arquitecturas compiladas: {' '.join(archs)}")
    print(f"GPU {arch} incluida en la build: {arch in archs}")
    try:
        x = torch.randn(512, 512, device="cuda")
        y = (x @ x).sum().item()
        print(f"Operación real en GPU: OK (checksum {y:.2f})")
    except Exception as exc:  # noqa: BLE001 - queremos reportar cualquier fallo
        print(f"Operación real en GPU: FALLÓ -> {exc}")


def main() -> None:
    report_system()
    report_gpu()
    report_ffmpeg()
    report_torch()


if __name__ == "__main__":
    main()
