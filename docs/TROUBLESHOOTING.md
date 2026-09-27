# Troubleshooting

Errores reales encontrados durante el desarrollo. Cada entrada incluye síntoma exacto, causa, solución y versiones.

Entorno de referencia: Windows 10 22H2 (19045), GTX 1060 6 GB (CC 6.1), driver 566.03, Python 3.11.4,
whisperx 3.8.6, torch 2.8.0+cu126, ctranslate2 4.8.2, faster-whisper 1.2.1, pyannote-audio 4.0.7,
huggingface-hub 0.36.2, torchcodec 0.7.0, ffmpeg 8.1.1 essentials (gyan.dev).

---

## CTranslate2: `Library cublas64_12.dll is not found or cannot be loaded`

- **Síntoma:** `RuntimeError: Library cublas64_12.dll is not found or cannot be loaded` al transcribir en `cuda`.
- **Causa:** la wheel de `ctranslate2` 4.8.2 para Windows solo trae `cudnn64_9.dll`; no incluye cuBLAS.
  La wheel `torch 2.8.0+cu126` sí trae `cublas64_12.dll` en `torch/lib`, y esa DLL se carga al hacer `import torch`.
- **Solución:** importar `torch` antes de crear el modelo de faster-whisper/whisperx. Verificado: con `import torch`
  previo, `int8` en `cuda` funciona. No hace falta instalar el CUDA Toolkit.

## CTranslate2: float16 no soportado en Pascal

- **Síntoma:** `ValueError: Requested float16 compute type, but the target device or backend do not support efficient float16 computation.`
- **Causa:** en CC 6.1, `ctranslate2.get_supported_compute_types("cuda")` devuelve solo `float32`, `int8` e `int8_float32`.
  whisperx usa `float16` por defecto en GPU (`whisperx/asr.py:351`).
- **Solución:** pasar `compute_type="int8"` (o `float32`) explícitamente.

## Hugging Face: `OSError: [WinError 1314] El cliente no dispone de un privilegio requerido` al descargar modelos

- **Síntoma:** la descarga de `Systran/faster-whisper-large-v3` falla en `huggingface_hub/file_download.py`,
  en `_create_symlink` → `os.symlink(...)`, con `WinError 1314`. Con el modelo `tiny` no falló (es intermitente).
- **Causa:** condición de carrera en `huggingface_hub` 0.36.2. `are_symlinks_supported()` guarda `True` en su caché
  *antes* de probar si puede crear un symlink (`file_download.py:108`), y `snapshot_download` descarga con 8 hilos.
  Los demás hilos leen ese `True` provisional e intentan crear symlinks, que en Windows requieren Developer Mode
  o permisos de administrador. La comprobación se hace por carpeta de repo, así que no basta con "calentar" la caché global.
- **Solución (aplicada en código):** al inicio del proceso, detectar una vez y fijar el resultado:
  ```python
  import huggingface_hub.file_download as hf_fd
  _symlinks = hf_fd.are_symlinks_supported()
  hf_fd.are_symlinks_supported = lambda cache_dir=None: _symlinks
  ```
  Si una descarga quedó a medias, borrar su carpeta en `%USERPROFILE%\.cache\huggingface\hub\models--...`.
  Alternativa sin código: activar Developer Mode en Windows.

## Diarización: `GatedRepoError: 403 ... pyannote/speaker-diarization-community-1`

- **Síntoma:** `Access to model pyannote/speaker-diarization-community-1 is restricted and you are not in the authorized list.`
- **Causa:** el token es válido (un token inválido daría 401), pero la cuenta no aceptó las condiciones del modelo.
  whisperx 3.8.6 usa ese modelo por defecto (`whisperx/diarize.py:101`).
- **Solución:** con la cuenta dueña del token, entrar a https://huggingface.co/pyannote/speaker-diarization-community-1
  y aceptar las condiciones (`gated=auto`: el acceso es inmediato). Es un único repo: incluye segmentation,
  embedding y PLDA.

## Diarización: `403 Forbidden: Please enable access to public gated repositories in your fine-grained token settings`

- **Síntoma:** con la licencia ya aceptada, la descarga de `config.yaml` de `pyannote/speaker-diarization-community-1`
  sigue fallando con 403. El mensaje final que se ve es `LocalEntryNotFoundError: An error happened while trying to
  locate the file on the Hub...`, que es engañoso: el 403 real aparece más arriba en el traceback.
- **Causa:** el token es *fine-grained* y no tiene el permiso de leer repos gated
  (`HfApi().whoami()["auth"]["accessToken"]["fineGrained"]["canReadGatedRepos"] == False`).
- **Solución:** usar un token de tipo **Read** o, en el token fine-grained, activar
  *"Read access to contents of all public gated repos you can access"*.

## CUDA out of memory en ASR con `batch_size=8`

- **Síntoma:** `RuntimeError: CUDA failed with error out of memory` en `generate_segment_batched`
  (large-v3, int8, GTX 1060 6 GB).
- **Causa:** con `batch_size=8` el pico total medido llegó a 5960/6144 MiB en una corrida (con ~0.9 GB ocupados
  por el escritorio y los navegadores), y en la siguiente se desbordó.
- **Solución:** `batch_size=4`: pico total de 5180 MiB (base 325 MiB). El default de la herramienta debe ser ≤ 4 en 6 GB.

## Aviso (inofensivo): `torchcodec is not installed correctly so built-in audio decoding will fail`

- **Síntoma:** `UserWarning` de `pyannote/audio/core/io.py:48` al importar: `Could not load libtorchcodec ... libtorchcodec_core7.dll`.
- **Causa:** torchcodec 0.7.0 en Windows necesita las DLL *compartidas* de FFmpeg 4–7. La build "essentials"
  de gyan.dev (la que instala winget) es estática y además es la 8.x.
- **Impacto:** ninguno. whisperx decodifica con el `ffmpeg` CLI y le pasa a pyannote el audio en memoria
  (`{'waveform', 'sample_rate'}`), que es justamente la vía alternativa que sugiere el aviso. Verificado: con este aviso,
  la diarización completa con community-1 funciona.
