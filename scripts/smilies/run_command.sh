#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SINGULARITY="${SINGULARITY:-singularity}"
IMAGE="${IMAGE:-$REPO/containers/temporal-event-spiking.sif}"

# --cleanenv drops host variables: explicitly forward the selected GPU and thread limits.
RUNTIME_ENV=(env CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
  "OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}" "MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}")
if [[ -v CUDA_VISIBLE_DEVICES ]]; then
  RUNTIME_ENV+=("CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
fi

if [[ "${1:-}" == "--foreground" ]]; then
  shift
  cd "$REPO"
  exec "$SINGULARITY" exec --cleanenv --nv --bind "$REPO:/workspace" \
    --pwd /workspace "$IMAGE" "${RUNTIME_ENV[@]}" python -m etsr.cli "$@"
fi

[[ "$#" -ge 3 && "$2" == "--" ]] || {
  echo "Uso: CUDA_VISIBLE_DEVICES=N bash $0 SESSIONE -- COMANDO_CLI [ARGOMENTI...]" >&2
  exit 2
}
SESSION="$1"
shift 2
[[ "$SESSION" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Sessione non valida" >&2; exit 2; }
command -v "$SINGULARITY" >/dev/null
command -v screen >/dev/null
[[ -f "$IMAGE" ]] || { echo "Container mancante: $IMAGE" >&2; exit 1; }
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "Il worktree deve essere pulito: sincronizzare un commit prima del lancio." >&2
  exit 1
fi
if screen -ls 2>/dev/null | awk -v session="$SESSION" '
  $1 ~ /^[0-9]+\./ { name = $1; sub(/^[0-9]+\./, "", name); if (name == session) found = 1 }
  END { exit !found }
'; then
  echo "Sessione già presente: $SESSION" >&2
  exit 1
fi

"$SINGULARITY" exec --cleanenv --nv "$IMAGE" "${RUNTIME_ENV[@]}" python -c \
  'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1, "Selezionare una sola GPU con CUDA_VISIBLE_DEVICES"; print(torch.cuda.get_device_name(0))'

LOG_FILE="$REPO/artifacts/screen/$SESSION.log"
mkdir -p "$(dirname -- "$LOG_FILE")"
screen -L -Logfile "$LOG_FILE" -dmS "$SESSION" \
  bash "$SCRIPT_DIR/run_command.sh" --foreground "$@"
echo "Avviato: $SESSION | log: $LOG_FILE | rientro: screen -r $SESSION"
