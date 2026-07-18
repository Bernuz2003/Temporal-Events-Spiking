#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/phase1_dvsgc_order2.yaml}"
python -m etsr.cli train --config "$CONFIG"
