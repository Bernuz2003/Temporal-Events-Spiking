#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SINGULARITY="${SINGULARITY:-singularity}"
IMAGE="${IMAGE:-$REPO/containers/temporal-event-spiking.sif}"
DEFINITION="$REPO/containers/temporal_event_spiking.def"

export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-$REPO/.singularity/cache}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-$REPO/.singularity/tmp}"

command -v "$SINGULARITY" >/dev/null 2>&1 || {
  echo "Comando Singularity non trovato: $SINGULARITY" >&2
  exit 1
}
[[ -f "$DEFINITION" ]] || {
  echo "Definition file mancante: $DEFINITION" >&2
  exit 1
}

mkdir -p "$(dirname -- "$IMAGE")" "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

build_options=(build --fakeroot)
if [[ -e "$IMAGE" ]]; then
  if [[ "${REBUILD:-0}" != "1" ]]; then
    echo "Immagine già presente: $IMAGE" >&2
    echo "Usare REBUILD=1 per ricostruirla esplicitamente." >&2
    exit 2
  fi
  build_options+=(--force)
fi

"$SINGULARITY" "${build_options[@]}" "$IMAGE" "$DEFINITION"
[[ -f "$IMAGE" ]] || {
  echo "La build è terminata senza produrre l'immagine attesa: $IMAGE" >&2
  exit 1
}
"$SINGULARITY" test "$IMAGE"

echo "Container verificato: $IMAGE"
