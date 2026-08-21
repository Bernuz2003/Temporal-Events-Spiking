#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/smoke.yaml}"
PYTHON="${PYTHON:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

exec "$PYTHON" -m etsr.cli smoke --config "$CONFIG"
