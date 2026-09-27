# Backlog

Ideas acordadas para después de v1. No están implementadas.

## Renombrar hablantes en una transcripción ya hecha

Pedido por Giano el 2026-09-27. Antes estaba fuera de alcance ("mapear SPEAKER_00 a nombres reales"); ahora va al final.

- Un comando aparte que no vuelve a transcribir. Toma una transcripción existente y cambia `SPEAKER_XX` por nombres.
- Debe ser cómodo en consola para reuniones de 2 a 6 hablantes (a veces más). Por ejemplo, mostrar 2 o 3 frases
  representativas de cada hablante y preguntar el nombre, con Enter para dejarlo como está. Pero también dar la opción de hacerlo si ya se tiene identificado quién es quién ágilmente.
- Debe permitir asignar el mismo nombre a dos etiquetas: pyannote a veces divide a una persona en dos
  (visto en la reunión de 69 min: SPEAKER_02 y SPEAKER_03 eran la misma persona). Después de unirlas hay que
  volver a fusionar los turnos consecutivos.
- Lo más robusto es partir del `.json`, que conserva segmentos, palabras y hablantes.

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
