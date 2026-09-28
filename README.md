# transcriptor-windows-whisperx

Transcribe grabaciones largas de reuniones 100% en local con [WhisperX](https://github.com/m-bain/whisperX):
transcripción (Whisper large-v3), timestamps por palabra y separación de hablantes (pyannote).

Verificado en Windows 10 con una GTX 1060 6 GB y con `--device cpu` (i5-8400). Lo marcado *(no verificado)* no se probó.

## Requisitos

- Windows 10 (Windows 11 *(no verificado)*), PowerShell y Git.
- Opcional: GPU NVIDIA de la serie GTX 10xx a RTX 40xx, con un driver que soporte CUDA 12.6 o superior
  (`nvidia-smi` lo muestra como "CUDA Version"). Si la GPU no es compatible, el programa avisa y usa la CPU, que
  es mucho más lenta. En una PC **sin** GPU NVIDIA *(no verificado)* debería pasar lo mismo. Solo se probó la CPU
  con `--device cpu`, en una PC que sí tiene GPU. Para RTX 50xx ver [más abajo](#gpu-rtx-50xx-no-verificado).
- Unos 15 GB libres: ~7 GB para el entorno de Python, ~3,5 GB para los modelos (se descargan en la primera
  transcripción) y la caché de descargas de uv.

No hace falta instalar Python: uv descarga Python 3.11 si no lo encuentra.

## Instalación

1. Instala [uv](https://docs.astral.sh/uv/) y [ffmpeg](https://ffmpeg.org/). Abre una terminal **nueva** después,
   para que tome el `PATH` actualizado:

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   winget install Gyan.FFmpeg.Essentials
   ```

2. Descarga el proyecto e instala sus dependencias (descarga ~3 GB):

   ```powershell
   git clone <url-del-repo>
   cd transcriptor-windows-whisperx
   uv sync
   ```

3. Comprueba el entorno:

   ```powershell
   uv run python scripts/diagnose.py
   ```

   Con GPU debe decir `torch: 2.8.0+cu126`, `GPU sm_XX incluida en la build: True` y `Operación real en GPU: OK`.
   Si dice `cuda.is_available(): False`, torch no ve la GPU y el programa usará la CPU.

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

### Poner nombres a los hablantes

También desde el `.json`, sin volver a transcribir. Sin `--names`, muestra las 3 frases más largas de cada
hablante (con su minuto) y pregunta su nombre. Enter deja el actual y `-` vuelve a la etiqueta original:

```powershell
uv run python -m transcriptor.rename outputs\reunion
```

Si ya sabes quién es quién:

```powershell
uv run python -m transcriptor.rename outputs\reunion --names "SPEAKER_00=Ana,SPEAKER_01=Luis,SPEAKER_02=Luis"
```

- Dar el mismo nombre a dos etiquetas las une (pyannote a veces divide a una persona en dos), y los turnos
  seguidos de esa persona se vuelven a fusionar.
- Se puede cambiar un nombre ya puesto (`--names "Luis=Luis Pérez"`) o quitarlo (`--names "SPEAKER_01="`).
- Los nombres se guardan en el `.json` (`meta.speaker_names`) sin borrar las etiquetas originales, y
  `transcriptor.render` los sigue usando. Volver a transcribir el audio los borra.
- Acepta `--timestamps` y `--formats` igual que `transcriptor.render`.

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

### Número de hablantes

Por defecto lo estima pyannote. A veces divide a una persona en dos etiquetas (`SPEAKER_02` y `SPEAKER_03`).
Fijarlo con `--speakers N` puede empeorar el resultado: en una reunión de 69 min con 3 personas, `--speakers 3`
unió esas dos etiquetas, pero le atribuyó a una persona frases de las otras en toda la reunión, incluso antes de
que llegara. Un rango (`--min-speakers` / `--max-speakers`) no cambia nada si la estimación ya cae dentro.

### Nombres y siglas: `--prompt` (con riesgo)

Whisper suele escribir mal los nombres propios y las siglas. `--prompt` le da un texto de contexto:

```powershell
uv run python -m transcriptor samples\reunion.mp3 --prompt "Hablamos de DSC PUCP, la AEE PUCP, IEEE CS PUCP y Don Keynesio."
```

Úsalo solo cuando los nombres importen más que el riesgo. En la reunión de 69 min de prueba, ese prompt corrigió
todas las apariciones de "Don Keynesio" y de "DSC", pero:

- perdió el 3,6 % de las palabras: en algunos trozos se saltó frases o reemplazó ~45 palabras por "¡Suscríbete al canal!";
- insertó siglas que nadie dijo (por ejemplo, "desde ese punto" pasó a "DSC PUCP").

Sin `--prompt` no pasa nada de esto. El prompt usado queda registrado en el `.json`.

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
La misma reunión, desde un clon limpio siguiendo este README (uv 0.12.19, ffmpeg 9.0.1), tardó 14 min 04 s y dio
un `.txt` idéntico.

En CPU (`--device cpu`, en la misma máquina), un audio de 5 min sin identificar hablantes tardó 4 min 27 s, y uno
de 45 s con hablantes tardó 1 min 18 s.

## GPU RTX 50xx *(no verificado)*

La build de torch que instala `uv sync` (cu126) incluye kernels para `sm_61` a `sm_90` (de GTX 10xx a RTX 40xx).
Las RTX 50xx (`sm_120`) no están incluidas: el programa debería detectarlo, avisar y usar la CPU.

Para intentar usar la GPU, cambia `cu126` por `cu128` en las dos líneas del índice `pytorch-cu126` de
`pyproject.toml` (`name` y `url`) y en las tres de `[tool.uv.sources]`, y reinstala:

```powershell
uv lock
uv sync
```

Nadie lo probó todavía. La build cu128 descarga ~3,5 GB y necesita un driver reciente (CUDA 12.8 o superior en
`nvidia-smi`). Tampoco se sabe si CTranslate2 4.8.2 (el motor de transcripción) funciona en `sm_120`.

## Problemas conocidos

Ver [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).
