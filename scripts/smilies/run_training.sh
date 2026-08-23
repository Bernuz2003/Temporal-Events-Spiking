#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
SINGULARITY="${SINGULARITY:-singularity}"
IMAGE="${IMAGE:-$REPO/containers/temporal-event-spiking.sif}"

usage() {
  echo "Uso: $0 CONFIG [SESSIONE]" >&2
  echo "Avvia in screen il training descritto da un file YAML interno al repository." >&2
}

resolve_config() {
  local requested="$1"
  local absolute
  if [[ "$requested" = /* ]]; then
    absolute="$(realpath -e -- "$requested")"
  else
    absolute="$(realpath -e -- "$REPO/$requested")"
  fi
  case "$absolute" in
    "$REPO"/*) printf '%s\n' "${absolute#"$REPO"/}" ;;
    *)
      echo "La configurazione deve trovarsi nel repository montato: $absolute" >&2
      exit 2
      ;;
  esac
}

run_foreground() {
  local config="$1"
  cd "$REPO"
  exec "$SINGULARITY" exec --cleanenv --nv \
    --bind "$REPO:/workspace" \
    --pwd /workspace "$IMAGE" \
    env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    python -m etsr.cli train --config "$config"
}

if [[ "${1:-}" == "--foreground" ]]; then
  [[ "$#" -eq 2 ]] || {
    usage
    exit 2
  }
  run_foreground "$2"
fi

[[ "$#" -ge 1 && "$#" -le 2 ]] || {
  usage
  exit 2
}
command -v "$SINGULARITY" >/dev/null 2>&1 || {
  echo "Comando Singularity non trovato: $SINGULARITY" >&2
  exit 1
}
command -v screen >/dev/null 2>&1 || {
  echo "Comando GNU Screen non trovato sull'host." >&2
  exit 1
}
[[ -f "$IMAGE" ]] || {
  echo "Container mancante: $IMAGE; eseguire make smilies-build." >&2
  exit 1
}

CONFIG="$(resolve_config "$1")"
SESSION="${2:-$(basename -- "${CONFIG%.yaml}")}"
[[ "$SESSION" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo "Nome sessione non valido: $SESSION" >&2
  exit 2
}
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "Il worktree non è pulito; il training non sarebbe riproducibile." >&2
  git -C "$REPO" status --short >&2
  exit 1
fi
if screen -ls 2>/dev/null | awk -v session="$SESSION" '
  $1 ~ /^[0-9]+\./ {
    name = $1
    sub(/^[0-9]+\./, "", name)
    if (name == session) found = 1
  }
  END { exit !found }
'; then
  echo "Esiste già una sessione screen chiamata $SESSION." >&2
  exit 1
fi

"$SINGULARITY" exec --cleanenv --nv "$IMAGE" \
  python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

LOG_DIR="$REPO/artifacts/screen"
LOG_FILE="$LOG_DIR/$SESSION.log"
mkdir -p "$LOG_DIR"
screen -L -Logfile "$LOG_FILE" -dmS "$SESSION" \
  bash "$SCRIPT_PATH" --foreground "$CONFIG"

echo "Training avviato nella sessione: $SESSION"
echo "Log screen: $LOG_FILE"
echo "Rientro: screen -r $SESSION"
echo "Monitoraggio: tail -f '$LOG_FILE'"
