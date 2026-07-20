#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/mechanistic_audit_dvsgc_order2.yaml}"
PYTHON="${PYTHON:-python}"

exec "$PYTHON" -m etsr.cli prepare-matched-dvsgc --config "$CONFIG"
