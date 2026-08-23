#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SINGULARITY="${SINGULARITY:-singularity}"
IMAGE="${IMAGE:-$REPO/containers/temporal-event-spiking.sif}"
TRAIN_ROOT="data/DVS-Lip/train"
SPLIT_MANIFEST="data/dvslip_development_split.json"

usage() {
  echo "Uso: $0 {prepare|gate}" >&2
}

require_runtime() {
  command -v "$SINGULARITY" >/dev/null 2>&1 || {
    echo "Comando Singularity non trovato: $SINGULARITY" >&2
    exit 1
  }
  [[ -f "$IMAGE" ]] || {
    echo "Container mancante: $IMAGE; eseguire make smilies-build." >&2
    exit 1
  }
  [[ -d "$REPO/$TRAIN_ROOT" ]] || {
    echo "Dataset train mancante: $REPO/$TRAIN_ROOT" >&2
    exit 1
  }
}

require_split() {
  [[ -f "$REPO/$SPLIT_MANIFEST" ]] || {
    echo "Split mancante: $REPO/$SPLIT_MANIFEST; eseguire prima prepare." >&2
    exit 1
  }
}

require_clean_worktree() {
  if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
    echo "Il worktree deve essere pulito prima del gate." >&2
    git -C "$REPO" status --short >&2
    exit 1
  fi
}

container() {
  local use_gpu="$1"
  shift
  local options=(exec --cleanenv)
  if [[ "$use_gpu" == "gpu" ]]; then
    options+=(--nv)
  fi
  "$SINGULARITY" "${options[@]}" \
    --bind "$REPO:/workspace" \
    --pwd /workspace "$IMAGE" "$@"
}

verify_artifacts() {
  container cpu python -c '
import json
from pathlib import Path

preflight = json.loads(Path("artifacts/dvslip_preflight.json").read_text())
shortcut = json.loads(Path("artifacts/dvslip_shortcut_control.json").read_text())
assert preflight["validation_status"] == "passed"
assert preflight["preflight_gate_status"] == "ready"
assert preflight["protocol_gate_status"] == "ready"
assert preflight["protocol_blockers"] == []
assert preflight["dataset_content"]["complete"] is True
assert preflight["official_test_used"] is False
assert shortcut["control_id"] == "dvslip_global_shortcuts_v1"
assert shortcut["official_test_used"] is False
print("Artifact gate verificato: ready, train-only")
'
}

prepare() {
  require_runtime
  container cpu python -m etsr.cli prepare-dvslip-split \
    --train-root "$TRAIN_ROOT" \
    --output "$SPLIT_MANIFEST"
}

gate() {
  require_runtime
  require_split
  require_clean_worktree
  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "nvidia-smi non disponibile sull'host." >&2
    exit 1
  }

  nvidia-smi
  container gpu python -c \
    'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
  container gpu make test PYTHON=python
  container cpu make lint RUFF=ruff
  container cpu make check-scripts
  container cpu python -m compileall -q src tests
  git -C "$REPO" diff --check
  container cpu make preflight-dvslip PYTHON=python \
    DVSLIP_TRAIN_ROOT="$TRAIN_ROOT" \
    DVSLIP_PREFLIGHT_OUTPUT=artifacts/dvslip_preflight.json \
    DVSLIP_HASH_SAMPLES=1
  container cpu make shortcut-dvslip PYTHON=python \
    DVSLIP_SHORTCUT_OUTPUT=artifacts/dvslip_shortcut_control.json
  verify_artifacts
  echo "Gate DVS-Lip completato."
}

case "${1:-}" in
  prepare) prepare ;;
  gate) gate ;;
  *) usage; exit 2 ;;
esac
