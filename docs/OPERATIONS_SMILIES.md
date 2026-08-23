# SMILIES operations

**Status:** DVS-Lip procedure defined; Singularity build and CUDA execution still require server
verification.

## Storage and process model

The SMILIES tutorials distinguish the network home from the server-local filesystem. Keep the Git
clone in the network home, but put the SIF, dataset, logs, artifacts and checkpoints under
`/home/users/$USER/etsr`. Fakeroot and `/home/users/$USER` must first be enabled by a SMILIES
administrator. Copy retained results back to persistent storage and free local space after the run.

`screen` runs on the host, outside Singularity. The SIF is immutable and receives the host NVIDIA
driver through `--nv`; no writable sandbox or Singularity instance is needed for one training job.

## Server paths and build

From the repository clone:

```bash
export REPO="$HOME/temporal-event-spiking-research"
export LOCAL_ROOT="/home/users/$USER/etsr"
export IMAGE="$LOCAL_ROOT/containers/temporal-event-spiking.sif"
export SINGULARITY_CACHEDIR="$LOCAL_ROOT/singularity/cache"
export SINGULARITY_TMPDIR="$LOCAL_ROOT/singularity/tmp"

mkdir -p "$LOCAL_ROOT"/{containers,data,artifacts,checkpoints,logs,configs} \
  "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
test -w "$LOCAL_ROOT"
singularity build --fakeroot "$IMAGE" \
  "$REPO/containers/temporal_event_spiking.def"
```

The image contains the Python dependencies, while the repository is mounted read/write at
`/workspace` and imported through `/workspace/src`. Do not run `pip install -e .` at execution time:
the immutable image does not need it, and user-site packages are deliberately disabled.

## Data layout

Git excludes all dataset files and generated split manifests. The local server tree must be:

```text
/home/users/$USER/etsr/data/
├── DVS-Lip/train/
└── dvslip_development_split.json
```

Generate the deterministic split in the container after the official `train/` directory is in
place:

```bash
singularity exec --cleanenv \
  --bind "$REPO:/workspace" \
  --bind "$LOCAL_ROOT/data:/workspace/data" \
  --pwd /workspace "$IMAGE" \
  python -m etsr.cli prepare-dvslip-split \
    --train-root data/DVS-Lip/train \
    --output data/dvslip_development_split.json
```

The official `test/` directory is not mounted or used during development.

## Pre-run gate

Run from a clean, pushed commit and verify the GPU in the same SIF:

```bash
cd "$REPO"
git status --short --branch
nvidia-smi

singularity exec --cleanenv --nv "$IMAGE" \
  python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); assert torch.cuda.is_available()'

singularity exec --cleanenv --nv \
  --bind "$REPO:/workspace" \
  --bind "$LOCAL_ROOT:/local" \
  --bind "$LOCAL_ROOT/data:/workspace/data:ro" \
  --pwd /workspace "$IMAGE" \
  make test PYTHON=python

singularity exec --cleanenv \
  --bind "$REPO:/workspace" \
  --bind "$LOCAL_ROOT:/local" \
  --bind "$LOCAL_ROOT/data:/workspace/data:ro" \
  --pwd /workspace "$IMAGE" \
  make preflight-dvslip PYTHON=python \
    DVSLIP_TRAIN_ROOT=data/DVS-Lip/train \
    DVSLIP_PREFLIGHT_OUTPUT=/local/artifacts/dvslip_preflight.json \
    DVSLIP_HASH_SAMPLES=1

singularity exec --cleanenv \
  --bind "$REPO:/workspace" \
  --bind "$LOCAL_ROOT:/local" \
  --bind "$LOCAL_ROOT/data:/workspace/data:ro" \
  --pwd /workspace "$IMAGE" \
  make shortcut-dvslip PYTHON=python \
    DVSLIP_SHORTCUT_OUTPUT=/local/artifacts/dvslip_shortcut_control.json
```

The full r0 must not start if tests, CUDA, preflight or the D016 shortcut control fail.

## One-epoch cost gate

Create a local pilot configuration without modifying the candidate tracked by Git:

```bash
sed \
  -e 's/^  name: dvslip_e0_recipe_r0$/  name: dvslip_e0_recipe_r0_cost_pilot/' \
  -e 's/^  recipe_id: dvslip_e0_r0$/  recipe_id: dvslip_e0_r0_cost_pilot/' \
  -e 's/^  epochs: 64$/  epochs: 1/' \
  -e 's/^  warmup_epochs: 4$/  warmup_epochs: 0/' \
  "$REPO/configs/dvslip_e0_recipe_r0.yaml" \
  > "$LOCAL_ROOT/configs/dvslip_e0_recipe_r0_cost_pilot.yaml"

screen -L -Logfile "$LOCAL_ROOT/logs/dvslip_e0_r0_cost_pilot.screen.log" \
  -S dvslip_e0_r0_cost_pilot
```

Inside that host `screen` session, run:

```bash
singularity exec --cleanenv --nv \
  --bind "$REPO:/workspace" \
  --bind "$LOCAL_ROOT:/local" \
  --bind "$LOCAL_ROOT/data:/workspace/data:ro" \
  --pwd /workspace "$IMAGE" \
  env ETSR_ARTIFACT_ROOT=/local/artifacts \
      ETSR_CHECKPOINT_ROOT=/local/checkpoints \
  python -m etsr.cli train \
    --config /local/configs/dvslip_e0_recipe_r0_cost_pilot.yaml
```

Inspect `epoch_seconds` in `history.csv` and `peak_cuda_memory_bytes` in `summary.json` before the
64-epoch commitment. Keep batch 4/accumulation 8 unless the measured cost makes r0 infeasible; a
micro-batch change also changes BatchNorm statistics and is therefore not an automatic throughput
tweak.

## DVS-Lip E0 training

Start a named, logged host session:

```bash
screen -L -Logfile "$LOCAL_ROOT/logs/dvslip_e0_r0.screen.log" -S dvslip_e0_r0
```

Inside `screen`, launch the immutable candidate configuration:

```bash
singularity exec --cleanenv --nv \
  --bind "$REPO:/workspace" \
  --bind "$LOCAL_ROOT:/local" \
  --bind "$LOCAL_ROOT/data:/workspace/data:ro" \
  --pwd /workspace "$IMAGE" \
  env ETSR_ARTIFACT_ROOT=/local/artifacts \
      ETSR_CHECKPOINT_ROOT=/local/checkpoints \
  python -m etsr.cli train --config configs/dvslip_e0_recipe_r0.yaml
```

Detach with `Ctrl-a`, then `d`; inspect with `screen -ls`; reattach with
`screen -r dvslip_e0_r0`. The run writes its own `run.log`, `environment.json`, resolved config,
history, metrics, diagnostics and summary below `/home/users/$USER/etsr/artifacts`, with checkpoints
in the parallel local directory.

## End of run

- confirm normal process termination and inspect both the screen log and run `summary.json`;
- verify `git_dirty=false`, the expected commit and dataset/split hashes in the run artifacts;
- copy compact retained artifacts and selected checkpoints to persistent storage;
- update the experiment ledger only if the result changes a scientific decision;
- remove disposable files from `/home/users/$USER` after confirming the copy.

Historical DVS-GC commands remain in
[`archive/dvsgc/smilies_setup.md`](archive/dvsgc/smilies_setup.md) for regression only.
