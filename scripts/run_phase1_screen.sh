#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-phase1_order2}"
IMAGE="${IMAGE:-/home/users/$USER/temporal-event-spiking.sif}"
REPO="${REPO:-$PWD}"
CONFIG="${CONFIG:-configs/phase1_dvsgc_order2.yaml}"
LOCAL_ROOT="${LOCAL_ROOT:-/home/users/$USER/etsr}"

mkdir -p "$LOCAL_ROOT/artifacts" "$LOCAL_ROOT/checkpoints"

COMMAND="export ETSR_ARTIFACT_ROOT=$LOCAL_ROOT/artifacts; \
export ETSR_CHECKPOINT_ROOT=$LOCAL_ROOT/checkpoints; \
singularity exec --nv \
  --bind $REPO:/workspace \
  --bind /home/users/$USER:/home/users/$USER \
  $IMAGE bash -lc 'cd /workspace && python -m pip install -e . --no-deps >/dev/null && python -m etsr.cli train --config $CONFIG'"

screen -dmS "$SESSION" bash -lc "$COMMAND"
echo "Sessione avviata: $SESSION"
echo "Controllo: screen -r $SESSION"
