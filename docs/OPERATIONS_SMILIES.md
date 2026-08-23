# SMILIES operations

**Status:** Singularity image and CUDA execution verified on `daredevil` (RTX A4000,
PyTorch 2.2.2+cu121); rerun the complete gate after the Ruff compatibility fix.

## Filesystem model

The current SMILIES installation no longer provides a separate `/home/users` area. The only project
root is:

```text
/home/ldapusers/z-tesisti/bernacchi/Temporal-Events-Spiking
```

The clone therefore contains code and all ignored runtime data:

```text
Temporal-Events-Spiking/
├── .singularity/                         # build cache and temporary files
├── artifacts/                            # metrics, logs and runtime pilot config
├── checkpoints/                          # model weights
├── containers/temporal-event-spiking.sif # immutable image
└── data/
    ├── DVS-Lip/train/
    └── dvslip_development_split.json
```

`screen` runs on the host. Singularity mounts the entire repository at `/workspace`, so relative
paths in YAML retain exactly the same meaning inside and outside the container. No writable
sandbox, instance, second bind or runtime `pip install` is needed.

## Synchronize and build

The server clone must contain the clean commit intended for the run:

```bash
cd /home/ldapusers/z-tesisti/bernacchi/Temporal-Events-Spiking
git switch developer
git pull --ff-only origin developer
git status --short --branch

make smilies-build
```

The build script uses `containers/temporal_event_spiking.def`, stores cache and temporary files in
`.singularity/`, builds `containers/temporal-event-spiking.sif` and runs `singularity test`. It
refuses to overwrite an existing image; rebuild deliberately with:

```bash
REBUILD=1 make smilies-build
```

## DVS-Lip sequence

Only the official `train/` directory is needed. Once it is present under `data/DVS-Lip/train`, run:

```bash
make smilies-dvslip-prepare
make smilies-dvslip-gate
make smilies-dvslip-pilot
```

`gate` verifies a clean worktree, host and container CUDA, pytest, Ruff, shell syntax, bytecode
compilation, the full hash preflight and the D016 shortcut control. `pilot` creates an ignored
one-epoch configuration without changing the versioned r0 recipe and starts it in the detached,
logged screen session `dvslip_e0_r0_cost_pilot`.

Inspect the latest pilot `history.csv` and `summary.json` under `artifacts/`. If runtime and peak CUDA
memory are acceptable, start the 64-epoch candidate:

```bash
make smilies-dvslip-train
```

The command refuses to start without preflight, shortcut and a clean completed CUDA pilot summary
from the current or immediately preceding operational-selection commit. It launches the versioned
`configs/dvslip_e0_recipe_r0.yaml` in session `dvslip_e0_r0` and writes screen logs to
`artifacts/screen/`.

## Screen controls

```bash
screen -ls
screen -r dvslip_e0_r0_cost_pilot
screen -r dvslip_e0_r0
tail -f artifacts/screen/dvslip_e0_r0.log
```

Detach with `Ctrl-a`, then `d`. A session terminates automatically when its training process exits.

## Generic future-dataset launcher

The SMILIES launcher does not encode a dataset name. Any configuration supported by the Python
pipeline can be started with:

```bash
make smilies-train \
  SMILIES_CONFIG=configs/<dataset-and-recipe>.yaml \
  SMILIES_SESSION=<descriptive-name>
```

This makes execution reusable without pretending that loaders for DailyDVS-200, DVS-Gesture or
CIFAR10-DVS already exist. Their Python dataset contracts will be integrated when each benchmark
becomes an active, source-verified task.

Historical DVS-GC helpers are isolated under `scripts/legacy/dvsgc/`; the old documentation remains
in [`archive/dvsgc/smilies_setup.md`](archive/dvsgc/smilies_setup.md) as provenance, not as the
current server procedure.
