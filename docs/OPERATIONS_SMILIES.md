# SMILIES operations

**Status:** Singularity image and CUDA execution verified on `daredevil` (RTX A4000,
PyTorch 2.2.2+cu121). Dataset preparation and gates use one dataset-driven workflow.

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
make smilies-prepare DATASET=dvslip
make smilies-gate DATASET=dvslip
```

`gate` verifies a clean worktree, host and container CUDA, pytest, Ruff, shell syntax, bytecode
compilation, the full hash preflight and the D016 shortcut control.

The corrected residual topology has passed its bounded overfit gate. The authorized complete E0
run uses the canonical configuration without overrides:

```bash
make smilies-train \
  SMILIES_CONFIG=configs/dvslip_e0.yaml \
  SMILIES_SESSION=dvslip_e0_full
```

## DVS-Gesture preparation

Manually extract the official `DvsGesture.tar.gz` so that
`data/DVS-Gesture/DvsGesture/trials_to_train.txt` exists. The active workflow deliberately reads
only the official train list:

```bash
make smilies-prepare DATASET=dvsgesture
make smilies-gate DATASET=dvsgesture
```

The first command writes derived raw-event segments under `data/DVS-Gesture/events/train`; the gate
exhaustively validates them and writes `artifacts/dvsgesture_dataset_profile.json`. If the extracted
directory differs, pass `DVSGESTURE_SOURCE_ROOT=path/inside/repository`.

## Screen controls

```bash
screen -ls
screen -r dvslip_e0_full
tail -f artifacts/screen/dvslip_e0_full.log
```

Detach with `Ctrl-a`, then `d`. A session terminates automatically when its training process exits.

## Generic future-dataset launcher

The SMILIES launcher does not encode a dataset name. Any configuration supported by the Python
pipeline can be started with:

```bash
make smilies-train \
  SMILIES_CONFIG=configs/<dataset-and-recipe>.yaml \
  SMILIES_SESSION=<descriptive-name> \
  SMILIES_TRAIN_ARGS='<optional train overrides>'
```

The DVS-Gesture adapter now uses this path. DailyDVS-200 and CIFAR10-DVS remain unimplemented until
each becomes an active, source-verified task.

Historical DVS-GC helpers are no longer part of the active runtime. Their documentation remains in
[`archive/dvsgc/smilies_setup.md`](archive/dvsgc/smilies_setup.md) as provenance, not as the current
server procedure; the executable implementation is recoverable from Git history.
