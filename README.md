# transcriptor-windows-whisperx

Transcribe grabaciones largas de reuniones 100% en local con [WhisperX](https://github.com/m-bain/whisperX):
transcripción (Whisper large-v3), timestamps por palabra y separación de hablantes (pyannote).

> Estado: Fase 1. Verificado en Windows 10 + GTX 1060 6 GB. Lo marcado *(no verificado)* aún no se probó.

## Requisitos

- Windows 10/11 y PowerShell.
- Python 3.11–3.13.
- [uv](https://docs.astral.sh/uv/) *(instalación de uv no verificada en esta guía)*.
- ffmpeg en `PATH` *(no verificado: `winget install Gyan.FFmpeg`; en la máquina de prueba ya estaba instalado así)*.
- Opcional: GPU NVIDIA con driver que soporte CUDA 12.6 o superior. Sin GPU funciona en CPU, más lento.
- Unos 10 GB libres para los modelos, que se descargan la primera vez.

Comprueba tu entorno con:

```powershell
python scripts/diagnose.py
```

## Instalación

```powershell
git clone <url-del-repo>
cd transcriptor-windows-whisperx
uv sync
```

Instala torch 2.8.0 **cu126**, la última serie con soporte para GPUs Pascal (GTX 10xx). Comprueba que quedó bien:

```powershell
uv run python scripts/diagnose.py
```

Debe decir `torch: 2.8.0+cu126`, `GPU sm_XX incluida en la build: True` y `Operación real en GPU: OK`.

## Hugging Face (solo para identificar hablantes)

1. Crea una cuenta en https://huggingface.co.
2. Acepta las condiciones de https://huggingface.co/pyannote/speaker-diarization-community-1 (el acceso es inmediato).
3. Crea un token de tipo **Read** en https://huggingface.co/settings/tokens.
   Si usas un token *fine-grained*, activa *"Read access to contents of all public gated repos you can access"*.
4. Copia `.env.example` como `.env` y pega el token:

   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

`.env` está en `.gitignore` y nunca se sube al repo.

## Uso

```powershell
uv run python -m transcriptor samples\reunion.mp3
```

Por defecto usa large-v3, español, GPU si está disponible e identificación de hablantes. Genera en `outputs\`:

- `reunion.txt`: texto legible, `[hh:mm:ss] SPEAKER_00: ...`, con turnos consecutivos del mismo hablante fusionados.
- `reunion.srt`: subtítulos.
- `reunion.json`: segmentos, palabras con timestamps y hablante, y metadatos de la corrida (tiempos por etapa).

Ejemplos:

```powershell
# Número exacto de hablantes, o un rango
uv run python -m transcriptor samples\reunion.m4a --speakers 3
uv run python -m transcriptor samples\reunion.m4a --min-speakers 2 --max-speakers 5

# Sin identificar hablantes, solo .txt, en otra carpeta
uv run python -m transcriptor samples\reunion.mp4 --no-diarize --formats txt --output-dir C:\transcripciones

# Forzar CPU
uv run python -m transcriptor samples\reunion.mp3 --device cpu

# Todas las opciones
uv run python -m transcriptor --help
```

Los valores por defecto están en `config.toml`. Cualquier opción del CLI los sobreescribe.

Si se pidió identificar hablantes y falla, el programa termina con error (código 3) y no genera salidas.
Nunca entrega en silencio una transcripción sin hablantes.

## Tiempos medidos

GTX 1060 6 GB, large-v3, int8, batch 4, audio de 5 min con 2 personas, modelos ya descargados:

| Etapa | Tiempo |
|---|---|
| Transcripción | 1 min 02 s |
| Alineación | 7,6 s |
| Hablantes | 15,8 s |
| **Total** | **1 min 26 s** |

En CPU, un audio de 45 s tardó 1 min 18 s en total.

## Problemas conocidos

Ver [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).
