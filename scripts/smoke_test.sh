#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

pytest -q
python - <<'PY'
from etsr.config import load_config
from etsr.runner import run_temporal_audit, train_experiment

config = load_config("configs/smoke.yaml")
summary = train_experiment(config)
run_temporal_audit(config, summary["checkpoint"])
PY
