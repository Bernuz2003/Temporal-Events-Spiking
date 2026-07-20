#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/mechanistic_audit_dvsgc_order2.yaml}"
SEED="${2:?Uso: scripts/train_audit_seed.sh [CONFIG] SEED}"
PYTHON="${PYTHON:-python}"

exec "$PYTHON" -m etsr.cli train --config "$CONFIG" --seed "$SEED"
