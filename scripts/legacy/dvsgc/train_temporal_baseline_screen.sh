#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/../../smilies/run_training.sh"
CONFIG="${CONFIG:-configs/temporal_audit_dvsgc_order2.yaml}"
SESSION="${SESSION:-temporal_audit_order2}"

exec bash "$RUNNER" "$CONFIG" "$SESSION"
