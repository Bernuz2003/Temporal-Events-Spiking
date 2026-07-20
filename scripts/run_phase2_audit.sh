#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/phase2_dvsgc_order2.yaml}"
shift || true
if [[ "$#" -lt 3 ]]; then
  echo "Uso: scripts/run_phase2_audit.sh CONFIG 42=PATH 123=PATH 2026=PATH" >&2
  exit 2
fi

PYTHON="${PYTHON:-python}"
ARGS=()
for checkpoint in "$@"; do
  ARGS+=(--checkpoint "$checkpoint")
done

exec "$PYTHON" -m etsr.cli phase2-audit --config "$CONFIG" "${ARGS[@]}"
