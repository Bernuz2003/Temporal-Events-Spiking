#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/phase2_dvsgc_order2.yaml}"
SEED="${2:?Uso: scripts/train_phase2_seed.sh [CONFIG] SEED}"
PYTHON="${PYTHON:-python}"

exec "$PYTHON" -m etsr.cli phase2-train --config "$CONFIG" --seed "$SEED"
