#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/phase2_dvsgc_order2.yaml}"
PYTHON="${PYTHON:-python}"

exec "$PYTHON" -m etsr.cli phase2-prepare --config "$CONFIG"
