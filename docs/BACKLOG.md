# Backlog

Ideas acordadas para después de v1. No están implementadas.

## Renombrar hablantes en una transcripción ya hecha

Pedido por Giano el 2026-09-27. Antes estaba fuera de alcance ("mapear SPEAKER_00 a nombres reales"); ahora va al final.

- Un comando aparte que no vuelve a transcribir. Toma una transcripción existente y cambia `SPEAKER_XX` por nombres.
- Debe ser cómodo en consola para reuniones de 2 a 6 hablantes (a veces más). Por ejemplo, mostrar 2 o 3 frases
  representativas de cada hablante y preguntar el nombre, con Enter para dejarlo como está.
- Debe permitir asignar el mismo nombre a dos etiquetas: pyannote a veces divide a una persona en dos
  (visto en la reunión de 69 min: SPEAKER_02 y SPEAKER_03 eran la misma persona). Después de unirlas hay que
  volver a fusionar los turnos consecutivos.
- Lo más robusto es partir del `.json`, que conserva segmentos, palabras y hablantes.
