#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SINGULARITY="${SINGULARITY:-singularity}"
IMAGE="${IMAGE:-$REPO/containers/temporal-event-spiking.sif}"
TRAIN_ROOT="data/DVS-Lip/train"
SPLIT_MANIFEST="data/dvslip_development_split.json"
R0_CONFIG="configs/dvslip_e0_recipe_r0.yaml"
PILOT_CONFIG="artifacts/runtime-configs/dvslip_e0_recipe_r0_cost_pilot.yaml"

usage() {
  cat >&2 <<EOF
Uso: $0 {prepare|gate|pilot|train}

  prepare  genera lo split development train-only
  gate     verifica CUDA, test, preflight completo e shortcut D016
  pilot    avvia in screen il cost pilot da un'epoca
  train    avvia in screen il training completo dvslip_e0_r0
EOF
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
    echo "Il worktree deve essere pulito prima di un gate o training." >&2
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

verify_gate_artifacts() {
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

verify_current_pilot() {
  local commit
  local parent_commit
  commit="$(git -C "$REPO" rev-parse HEAD)"
  parent_commit="$(git -C "$REPO" rev-parse HEAD^)"
  container cpu python - "$commit" "$parent_commit" <<'PY'
import json
import sys
from pathlib import Path

commit = sys.argv[1]
accepted_commits = {commit, sys.argv[2]}
matches = []
for summary_path in Path("artifacts").glob(
    "dvslip_e0_recipe_r0_cost_pilot__*/summary.json"
):
    summary = json.loads(summary_path.read_text())
    if (
        summary.get("git_commit") in accepted_commits
        and summary.get("git_dirty") is False
        and summary.get("official_test_used") is False
        and int(summary.get("peak_cuda_memory_bytes", 0)) > 0
    ):
        matches.append(summary_path)
if not matches:
    raise SystemExit(
        "Nessun cost pilot CUDA pulito trovato per il commit corrente o il suo parent."
    )
print(f"Cost pilot coerente: {sorted(matches)[-1]}")
PY
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
  verify_gate_artifacts

  echo "Gate DVS-Lip completato. Esaminare gli artifact prima del pilot."
}

pilot() {
  require_runtime
  require_split
  require_clean_worktree
  mkdir -p "$REPO/$(dirname -- "$PILOT_CONFIG")"
  sed \
    -e 's/^  name: dvslip_e0_recipe_r0$/  name: dvslip_e0_recipe_r0_cost_pilot/' \
    -e 's/^  recipe_id: dvslip_e0_r0$/  recipe_id: dvslip_e0_r0_cost_pilot/' \
    -e 's/^  epochs: 64$/  epochs: 1/' \
    -e 's/^  warmup_epochs: 4$/  warmup_epochs: 0/' \
    "$REPO/$R0_CONFIG" > "$REPO/$PILOT_CONFIG"
  exec bash "$SCRIPT_DIR/run_training.sh" "$PILOT_CONFIG" dvslip_e0_r0_cost_pilot
}

train() {
  require_runtime
  require_split
  require_clean_worktree
  [[ -f "$REPO/artifacts/dvslip_preflight.json" ]] || {
    echo "Preflight artifact mancante; eseguire prima gate." >&2
    exit 1
  }
  [[ -f "$REPO/artifacts/dvslip_shortcut_control.json" ]] || {
    echo "Shortcut artifact mancante; eseguire prima gate." >&2
    exit 1
  }
  verify_gate_artifacts
  verify_current_pilot
  exec bash "$SCRIPT_DIR/run_training.sh" "$R0_CONFIG" dvslip_e0_r0
}

case "${1:-}" in
  prepare) prepare ;;
  gate) gate ;;
  pilot) pilot ;;
  train) train ;;
  *) usage; exit 2 ;;
esac
