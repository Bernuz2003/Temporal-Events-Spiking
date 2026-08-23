#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/temporal_audit_dvsgc_order2.yaml}"
PYTHON="${PYTHON:-python}"

exec "$PYTHON" -m etsr.cli train --config "$CONFIG"
