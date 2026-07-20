#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/mechanistic_audit_dvsgc_order2.yaml}"
shift || true
if [[ "$#" -lt 3 ]]; then
  echo "Uso: scripts/run_mechanistic_audit.sh CONFIG 42=PATH 123=PATH 2026=PATH" >&2
  exit 2
fi

PYTHON="${PYTHON:-python}"
ARGS=()
for checkpoint in "$@"; do
  ARGS+=(--checkpoint "$checkpoint")
done

exec "$PYTHON" -m etsr.cli mechanistic-audit --config "$CONFIG" "${ARGS[@]}"
