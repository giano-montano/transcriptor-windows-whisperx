# Backlog

Ideas acordadas para después de v1. No están implementadas.

## Filtro de texto inventado en silencios o ruido

Evaluado en la Fase 2 (2026-09-27). Giano eligió no filtrar por ahora; puede interesar más adelante.

- En modo batch, whisperx 3.8.6 **ignora** `no_speech_threshold`, `log_prob_threshold`, `compression_ratio_threshold`
  y el fallback de temperatura: `generate_segment_batched` (`whisperx/asr.py:36`) hace un solo `generate` por lote.
- No sirven para detectarlo: `avg_logprob` (lo inventado tiene -0,10 a -0,29; la mediana de la reunión es -0,12),
  el score de alineación por palabra (el habla real también tiene valores bajos; p10 = 0,39), ni subir `vad_onset`
  a 0,6 o 0,7 (el VAD marca como habla trozos de ~20 s de ruido).
- `repetition_penalty=1.1` y `no_repeat_ngram_size=3` cortan el bucle pero inventan otro texto y cambian el texto real.
- Propuesta si se retoma: filtro reversible. El `.json` conserva todo y marca los segmentos sospechosos, y `render`
  los omite. Reglas: (1) "Gracias." suelto después de ≥10 s de silencio; (2) una frase repetida ≥4 veces en <30 s queda
  en una sola aparición. Antes de adoptarla hay que escuchar 39:04 y 51:53 de la reunión de 69 min (candidatos dudosos
  a la regla 1).

## Prompt con red de seguridad (opción C de la Fase 2)

Evaluado en la Fase 2 (2026-09-27). Giano eligió `--prompt` opcional por corrida (opción B) y dejó esta idea para después.

- Datos (reunión de 69 min, prompt *"Hablamos de DSC PUCP, la AEE PUCP, IEEE CS PUCP y Don Keynesio."*):
  - corrige Don Keynesio (11 de 11) y DSC (13 apariciones, contra 0 sin prompt);
  - pierde 358 palabras netas (3,6 %) en 12 de 177 trozos de VAD; en 24:25 y 67:17, ~45 palabras reales se cambian por
    "¡Suscríbete al canal!";
  - inserta términos: "DSC PUCP" en 07:23, 11:56 y 12:32, donde se dijo "desde ese punto" (Giano lo confirmó escuchando);
    "AEE" ×3 en 19:08; "PUCP" en una lista de números en 65:28.
- Por qué pasa: en whisperx batch, el mismo prompt va como "texto anterior" (`sot_prev`) en **cada** trozo de ~30 s
  (`whisperx/asr.py:47-57`, `faster_whisper/transcribe.py:1532`), sin `condition_on_previous_text`. En trozos con poca
  voz o ruido, el modelo sigue el prompt en vez del audio.
- Idea: transcribir dos veces, con y sin prompt (los trozos de VAD son idénticos, así que se comparan uno a uno), y en
  cada trozo usar la versión con prompt salvo que:
  1. pierda más del ~25 % de las palabras respecto de la versión sin prompt;
  2. contenga frases inventadas conocidas ("Suscríbete al canal", "Gracias por ver"...);
  3. meta términos del prompt que en la versión sin prompt no tienen nada fonéticamente parecido (lo más difícil).
- Costo: +1 ASR (~9 min por hora de audio en la GTX 1060). La alineación y la diarización se hacen una sola vez.
- Script de experimentos: `outputs/fase2/exp_asr.py` (no versionado).
