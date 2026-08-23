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
```

`gate` verifies a clean worktree, host and container CUDA, pytest, Ruff, shell syntax, bytecode
compilation, the full hash preflight and the D016 shortcut control.

The generic launcher accepts bounded training overrides. The current optimization gate uses 16
classes and four official-train samples per class, disables augmentation, AMP and profiling, and
reuses the same subset for training and evaluation:

```bash
make smilies-train \
  SMILIES_CONFIG=configs/dvslip_e0.yaml \
  SMILIES_SESSION=dvslip_e0_overfit \
  SMILIES_TRAIN_ARGS='--overfit 16 4 --epochs 50'
```

Do not start the complete run until that gate is reviewed. Once authorized, the same launcher is
used without overfit arguments and without a dataset-specific training script.

## Screen controls

```bash
screen -ls
screen -r dvslip_e0_overfit
tail -f artifacts/screen/dvslip_e0_overfit.log
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

This makes execution reusable without pretending that loaders for DailyDVS-200, DVS-Gesture or
CIFAR10-DVS already exist. Their Python dataset contracts will be integrated when each benchmark
becomes an active, source-verified task.

Historical DVS-GC helpers are isolated under `scripts/legacy/dvsgc/`; the old documentation remains
in [`archive/dvsgc/smilies_setup.md`](archive/dvsgc/smilies_setup.md) as provenance, not as the
current server procedure.
