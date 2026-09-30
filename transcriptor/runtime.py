"""Preparación del proceso en Windows y chequeos previos (antes de cargar modelos).

Cada ajuste corresponde a un problema real documentado en docs/TROUBLESHOOTING.md.
"""

from __future__ import annotations

import logging
import os
import shutil
import warnings


class SetupError(RuntimeError):
    """Problema de entorno detectado antes de empezar a transcribir."""


def setup() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    # Avisos inofensivos y conocidos. torchcodec no se usa: whisperx pasa el audio en memoria a pyannote.
    warnings.filterwarnings("ignore", message=r"\s*torchcodec is not installed correctly")
    warnings.filterwarnings("ignore", message=r".*TensorFloat-32 \(TF32\) has been disabled")
    warnings.filterwarnings("ignore", message=r"std\(\): degrees of freedom is <= 0")
    # pyannote carga checkpoints viejos y pytorch_lightning avisa en INFO que los actualizó en memoria.
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    # torch primero: carga cublas64_12.dll, que la wheel de ctranslate2 no trae en Windows.
    import torch  # noqa: F401

    # Race en huggingface_hub 0.36: detectar el soporte de symlinks una sola vez y fijar el resultado.
    import huggingface_hub.file_download as hf_fd

    symlinks = hf_fd.are_symlinks_supported()
    hf_fd.are_symlinks_supported = lambda cache_dir=None: symlinks

    import whisperx

    whisperx.setup_logging(level="warning")


def check_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise SetupError("ffmpeg no está en PATH. Instálalo (winget install Gyan.FFmpeg.Essentials) y abre una terminal nueva.")


def resolve_device(requested: str) -> tuple[str, str | None]:
    """Devuelve (dispositivo, aviso). Falla si se pidió cuda y no es utilizable."""
    import torch

    reason = None
    if not torch.cuda.is_available():
        reason = f"torch {torch.__version__} no detecta CUDA"
    else:
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
        if arch not in torch.cuda.get_arch_list():
            reason = f"la build torch {torch.__version__} no incluye kernels {arch} para esta GPU"

    if requested == "cpu":
        return "cpu", None
    if reason is None:
        return "cuda", None
    if requested == "cuda":
        raise SetupError(f"Se pidió --device cuda pero {reason}.")
    return "cpu", f"GPU no utilizable ({reason}); se usará CPU (mucho más lento)."


def resolve_compute_type(device: str, requested: str) -> str:
    """Devuelve el compute_type a usar. "auto": int8_float16 si la GPU lo soporta (RTX), si no int8 (Pascal)."""
    import ctranslate2

    supported = ctranslate2.get_supported_compute_types(device)
    if requested == "auto":
        return "int8_float16" if device == "cuda" and "int8_float16" in supported else "int8"
    if requested not in supported:
        raise SetupError(
            f"compute_type {requested!r} no está soportado en {device} por esta máquina. "
            f"Opciones: {', '.join(sorted(supported))}."
        )
    return requested


# VRAM que necesita la ASR con large-v3 en int8/int8_float16: base + costo por elemento del batch, con margen.
# Medido en una RTX 3050 Laptop 4 GB (docs/TROUBLESHOOTING.md): batch 2 cabe con 3,3 GB libres y batch 4 no.
ASR_BASE_MB, ASR_PER_BATCH_MB = 2300, 320
BATCH_CHOICES = (4, 2, 1)  # 4 es el máximo verificado en 6 GB; 8 desbordó una GTX 1060


def resolve_batch_size(device: str, requested: int | str) -> tuple[int, str | None]:
    """Devuelve (batch, aviso). "auto" elige el mayor de BATCH_CHOICES que cabe en la VRAM libre ahora mismo."""
    if requested != "auto":
        return int(requested), None
    if device != "cuda":
        return BATCH_CHOICES[0], None
    free_mb = free_vram_mb()
    for batch in BATCH_CHOICES:
        if ASR_BASE_MB + ASR_PER_BATCH_MB * batch <= free_mb:
            return batch, None
    return 1, (
        f"solo hay {free_mb} MiB libres en la GPU y large-v3 necesita ~{ASR_BASE_MB + ASR_PER_BATCH_MB} MiB. "
        "Cierra programas que usen la GPU, usa un modelo más chico (--model medium) o --device cpu."
    )


def free_vram_mb() -> int | None:
    import torch

    return torch.cuda.mem_get_info()[0] // 2**20 if torch.cuda.is_available() else None


def is_out_of_memory(exc: BaseException) -> bool:
    # CTranslate2: "CUDA failed with error out of memory"; torch: torch.cuda.OutOfMemoryError ("CUDA out of memory").
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc)


def check_diarization_access(model: str, token: str | None) -> None:
    """Verifica token y licencia antes de la ASR, para no fallar después de una hora de trabajo."""
    if os.path.isdir(model):
        return
    if not token:
        raise SetupError(
            "La diarización está activada pero falta HF_TOKEN (en .env o como variable de entorno). "
            "Ver README, sección Hugging Face. Usa --no-diarize para transcribir sin hablantes."
        )
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    try:
        hf_hub_download(model, "config.yaml", token=token)
    except (HfHubHTTPError, LocalEntryNotFoundError) as exc:
        # LocalEntryNotFoundError envuelve el 403/401 real: se busca la primera causa con respuesta HTTP.
        root = exc
        while getattr(root, "response", None) is None and root.__cause__ is not None:
            root = root.__cause__
        status = getattr(getattr(root, "response", None), "status_code", None)
        hints = {
            401: "el token no es válido o expiró.",
            403: (
                f"la cuenta no tiene acceso. Acepta las condiciones en https://huggingface.co/{model} "
                "y, si el token es fine-grained, activa 'Read access to contents of all public gated repos "
                "you can access' (o usa un token de tipo Read)."
            ),
        }
        detail = hints.get(status, str(root).strip().splitlines()[0])
        raise SetupError(f"No se puede descargar el modelo de diarización {model}: {detail}") from exc
