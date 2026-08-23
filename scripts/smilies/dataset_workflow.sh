#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SINGULARITY="${SINGULARITY:-singularity}"
IMAGE="${IMAGE:-$REPO/containers/temporal-event-spiking.sif}"
DVSLIP_TRAIN_ROOT="${DVSLIP_TRAIN_ROOT:-data/DVS-Lip/train}"
DVSLIP_SPLIT_MANIFEST="${DVSLIP_SPLIT_MANIFEST:-data/dvslip_development_split.json}"
DVSGESTURE_SOURCE_ROOT="${DVSGESTURE_SOURCE_ROOT:-data/DVS-Gesture/DvsGesture}"
DVSGESTURE_TRAIN_ROOT="${DVSGESTURE_TRAIN_ROOT:-data/DVS-Gesture/events/train}"

usage() {
  echo "Uso: $0 {dvslip|dvsgesture} {prepare|gate}" >&2
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
}

require_directory() {
  [[ -d "$REPO/$1" ]] || {
    echo "Directory mancante: $REPO/$1" >&2
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

common_gate() {
  require_runtime
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
}

prepare_dvslip() {
  require_runtime
  require_directory "$DVSLIP_TRAIN_ROOT"
  container cpu python -m etsr.cli prepare-dvslip-split \
    --train-root "$DVSLIP_TRAIN_ROOT" \
    --output "$DVSLIP_SPLIT_MANIFEST"
}

gate_dvslip() {
  require_directory "$DVSLIP_TRAIN_ROOT"
  [[ -f "$REPO/$DVSLIP_SPLIT_MANIFEST" ]] || {
    echo "Split mancante: $REPO/$DVSLIP_SPLIT_MANIFEST; eseguire prima prepare." >&2
    exit 1
  }
  common_gate
  container cpu make preflight-dvslip PYTHON=python \
    DVSLIP_TRAIN_ROOT="$DVSLIP_TRAIN_ROOT" \
    DVSLIP_PREFLIGHT_OUTPUT=artifacts/dvslip_preflight.json \
    DVSLIP_HASH_SAMPLES=1
  container cpu make shortcut-dvslip PYTHON=python \
    DVSLIP_SHORTCUT_OUTPUT=artifacts/dvslip_shortcut_control.json
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
print("Artifact gate DVS-Lip verificato: ready, train-only")
'
}

prepare_dvsgesture() {
  require_runtime
  require_directory "$DVSGESTURE_SOURCE_ROOT"
  container cpu python -m etsr.cli prepare-dvsgesture \
    --source-root "$DVSGESTURE_SOURCE_ROOT" \
    --output-root "$DVSGESTURE_TRAIN_ROOT" \
    --report artifacts/dvsgesture_preparation.json
}

gate_dvsgesture() {
  require_directory "$DVSGESTURE_TRAIN_ROOT"
  common_gate
  container cpu python -m etsr.cli profile-dvsgesture \
    --train-root "$DVSGESTURE_TRAIN_ROOT" \
    --output artifacts/dvsgesture_dataset_profile.json
  container cpu python -c '
import json
from pathlib import Path

profile = json.loads(Path("artifacts/dvsgesture_dataset_profile.json").read_text())
preparation = json.loads(Path("artifacts/dvsgesture_preparation.json").read_text())
assert preparation["source_split"] == "train"
assert preparation["official_test_used"] is False
assert profile["dataset"]["class_count"] == 11
assert profile["dataset"]["sample_count"] == preparation["sample_count"]
assert profile["validation"]["all_samples_valid"] is True
assert profile["official_test_used"] is False
print("Artifact gate DVS-Gesture verificato: train-only")
'
}

DATASET="${1:-}"
ACTION="${2:-}"
case "$DATASET:$ACTION" in
  dvslip:prepare) prepare_dvslip ;;
  dvslip:gate) gate_dvslip ;;
  dvsgesture:prepare) prepare_dvsgesture ;;
  dvsgesture:gate) gate_dvsgesture ;;
  *) usage; exit 2 ;;
esac
