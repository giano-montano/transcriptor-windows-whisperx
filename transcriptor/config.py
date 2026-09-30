"""Carga de configuración: config.toml versionado + sobreescrituras del CLI."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.toml"
FORMATS = ("txt", "srt", "json")


@dataclass
class Settings:
    model: str = "large-v3"
    language: str | None = "es"
    batch_size: int | str = "auto"  # "auto": según VRAM libre (runtime.resolve_batch_size)
    initial_prompt: str | None = None  # solo por corrida (--prompt); ver README
    device: str = "auto"
    compute_type_cuda: str = "auto"  # "auto": según la GPU (runtime.resolve_compute_type)
    compute_type_cpu: str = "int8"
    compute_type: str | None = None  # si se define, gana sobre compute_type_<device>
    diarize: bool = True
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    output_dir: Path = Path("outputs")
    formats: list[str] = field(default_factory=lambda: ["txt"])
    timestamps: bool = False

    def compute_type_for(self, device: str) -> str:
        if self.compute_type:
            return self.compute_type
        return self.compute_type_cuda if device == "cuda" else self.compute_type_cpu


def load_settings(path: Path | None = None) -> Settings:
    path = path or DEFAULT_CONFIG
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo de configuración: {path}")
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    asr, dev = raw.get("asr", {}), raw.get("device", {})
    dia, out = raw.get("diarization", {}), raw.get("output", {})
    values = {
        "model": asr.get("model"),
        "language": asr.get("language"),
        "batch_size": asr.get("batch_size"),
        "device": dev.get("device"),
        "compute_type_cuda": dev.get("compute_type_cuda"),
        "compute_type_cpu": dev.get("compute_type_cpu"),
        "diarize": dia.get("enabled"),
        "diarization_model": dia.get("model"),
        "num_speakers": dia.get("num_speakers"),
        "min_speakers": dia.get("min_speakers"),
        "max_speakers": dia.get("max_speakers"),
        "output_dir": Path(out["dir"]) if "dir" in out else None,
        "formats": out.get("formats"),
        "timestamps": out.get("timestamps"),
    }
    settings = Settings(**{k: v for k, v in values.items() if v is not None})
    validate(settings)
    return settings


def apply_overrides(settings: Settings, overrides: dict) -> Settings:
    names = {f.name for f in fields(Settings)}
    for key, value in overrides.items():
        if value is not None and key in names:
            setattr(settings, key, value)
    if overrides.get("num_speakers") is not None:
        # Un número exacto anula cualquier rango heredado del archivo de configuración.
        settings.min_speakers = settings.max_speakers = None
    elif overrides.get("min_speakers") is not None or overrides.get("max_speakers") is not None:
        settings.num_speakers = None
    validate(settings)
    return settings


def validate(s: Settings) -> None:
    if s.language in ("auto", ""):
        s.language = None
    if s.device not in ("auto", "cuda", "cpu"):
        raise ValueError(f"device debe ser auto, cuda o cpu (recibido: {s.device!r})")
    unknown = [f for f in s.formats if f not in FORMATS]
    if unknown or not s.formats:
        raise ValueError(f"Formatos no válidos: {unknown or s.formats}. Opciones: {', '.join(FORMATS)}")
    if isinstance(s.batch_size, str) and s.batch_size.isdigit():
        s.batch_size = int(s.batch_size)
    if s.batch_size != "auto" and (not isinstance(s.batch_size, int) or s.batch_size < 1):
        raise ValueError(f'batch_size debe ser "auto" o un entero >= 1 (recibido: {s.batch_size!r})')
    if s.num_speakers is not None and (s.min_speakers is not None or s.max_speakers is not None):
        raise ValueError("Usa num_speakers (exacto) o min/max_speakers (rango), no ambos.")
    for name in ("num_speakers", "min_speakers", "max_speakers"):
        v = getattr(s, name)
        if v is not None and v < 1:
            raise ValueError(f"{name} debe ser >= 1")
    if s.min_speakers and s.max_speakers and s.min_speakers > s.max_speakers:
        raise ValueError("min_speakers no puede ser mayor que max_speakers")
