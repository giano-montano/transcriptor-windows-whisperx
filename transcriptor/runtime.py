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
        raise SetupError("ffmpeg no está en PATH. Instálalo (winget install Gyan.FFmpeg) y abre una terminal nueva.")


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


def check_compute_type(device: str, compute_type: str) -> None:
    import ctranslate2

    supported = ctranslate2.get_supported_compute_types(device)
    if compute_type not in supported:
        raise SetupError(
            f"compute_type {compute_type!r} no está soportado en {device} por esta máquina. "
            f"Opciones: {', '.join(sorted(supported))}."
        )


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
