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

Por defecto usa large-v3, español, GPU si está disponible e identificación de hablantes. Cada audio tiene su
propia carpeta en `outputs\`, que se sobrescribe si ya existe:

```
outputs\reunion\
  reunion.txt    texto legible "SPEAKER_00: ...", con los turnos consecutivos del mismo hablante fusionados
  reunion.json   palabras con tiempo y hablante, más metadatos de la corrida (se guarda siempre)
```

Bajo pedido:

- `--timestamps`: agrega `[hh:mm:ss]` al inicio de cada turno del `.txt`.
- `--formats txt,srt`: agrega un `.srt` (subtítulos).

### Regenerar salidas sin volver a transcribir

Desde el `.json`, en menos de un segundo y sin GPU:

```powershell
# .txt con tiempos
uv run python -m transcriptor.render outputs\reunion --timestamps

# .txt sin tiempos, más .srt
uv run python -m transcriptor.render outputs\reunion --formats txt,srt
```

Acepta la carpeta o el `.json` y reemplaza los archivos que regenera.

Ejemplos:

```powershell
# Número exacto de hablantes, o un rango
uv run python -m transcriptor samples\reunion.m4a --speakers 3
uv run python -m transcriptor samples\reunion.m4a --min-speakers 2 --max-speakers 5

# Sin identificar hablantes, solo .txt, en otra carpeta
uv run python -m transcriptor samples\reunion.mp4 --no-diarize --output-dir C:\transcripciones

# Con tiempos en el .txt, más .srt
uv run python -m transcriptor samples\reunion.mp3 --timestamps --formats txt,srt

# Forzar CPU
uv run python -m transcriptor samples\reunion.mp3 --device cpu

# Todas las opciones
uv run python -m transcriptor --help
```

Los valores por defecto están en `config.toml`. Cualquier opción del CLI los sobreescribe.

Si se pidió identificar hablantes y falla, el programa termina con error (código 3) y no genera salidas.
Nunca entrega en silencio una transcripción sin hablantes.

## Tiempos medidos

GTX 1060 6 GB, large-v3, int8, batch 4, modelos ya descargados:

| Etapa | Reunión de 69 min | Audio de 5 min |
|---|---|---|
| Carga de audio | 5,6 s | 0,4 s |
| Transcripción | 8 min 43 s | 1 min 02 s |
| Alineación | 1 min 19 s | 7,6 s |
| Hablantes | 3 min 32 s | 15,8 s |
| **Total** | **13 min 39 s** | **1 min 26 s** |

Pico de VRAM en la reunión de 69 min: 5,6 GB de 6 GB, contando lo que ya usaban el escritorio y los navegadores.

En CPU, un audio de 45 s tardó 1 min 18 s en total.

## Problemas conocidos

Ver [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).
