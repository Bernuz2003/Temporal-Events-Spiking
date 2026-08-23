#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-data/dvsgc}"
DOWNLOAD_DIR="$ROOT/download"
mkdir -p "$DOWNLOAD_DIR"

cat <<MSG
Directory preparata: $DOWNLOAD_DIR

Scaricare manualmente dal dataset DVS128 Gesture i quattro file richiesti dal package DVS-Gesture-Chain:
  - DvsGesture.tar.gz
  - gesture_mapping.csv
  - LICENSE.txt
  - README.txt

e collocarli in:
  $DOWNLOAD_DIR

Il primo avvio creerà prima gli eventi NumPy e poi le sequenze DVS-GC integrate in frame.
Dopo aver verificato la corretta creazione di events_np, download/ ed extract/ possono essere rimossi
secondo le indicazioni del repository ufficiale, mantenendo una copia sicura dei dati originali.
MSG

missing=0
for file in DvsGesture.tar.gz gesture_mapping.csv LICENSE.txt README.txt; do
  if [[ ! -f "$DOWNLOAD_DIR/$file" ]]; then
    printf 'MANCANTE: %s\n' "$DOWNLOAD_DIR/$file"
    missing=1
  fi
done

if [[ "$missing" -eq 0 ]]; then
  echo "Tutti i file richiesti sono presenti."
  exit 0
fi

if find "$ROOT/events_np" -type f -name '*.npz' -print -quit 2>/dev/null | grep -q .; then
  echo "I file originali non sono più necessari: events_np contiene già gli eventi estratti."
  exit 0
fi

if find "$ROOT" -type f -name '*.npz' -path "$ROOT/DVSGC_frames_number_*/*" -print -quit 2>/dev/null | grep -q .; then
  echo "I file originali non sono necessari per le configurazioni DVS-GC già integrate presenti."
  echo "Serviranno nuovamente per generare configurazioni non ancora presenti se manca events_np."
  exit 0
fi

echo "Preparazione incompleta: i quattro file sono indispensabili per la prima generazione." >&2
exit 1
