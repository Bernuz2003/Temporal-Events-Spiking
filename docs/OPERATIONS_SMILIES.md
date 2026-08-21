# SMILIES operations

**Status:** procedure prepared from the legacy workflow; execution not verified in this local runtime

## Preconditions

Before a server run record:

```text
branch: developer
commit:
dirty state:
config and recipe ID:
dataset/split hashes:
artifact/checkpoint roots:
```

The current repository has only a GitHub `origin`, while the charter refers to SMILIES GitLab. Do
not push `developer` or the closeout tag until P0-13 identifies the intended integration remote and
confirms whether GitHub is a mirror or the actual destination.

Run the quality gate in the same container that will launch the job. Inspect `nvtop` and `htop`, use
a named `screen` session, and verify that the artifact directory is writable before detaching.

## Build

The current definition remains `containers/temporal_event_spiking.def` and was designed for the
legacy frame-first environment. On SMILIES:

```bash
cd /home/users/<username>
singularity build --fakeroot temporal-event-spiking.sif \
  /path/to/repository/containers/temporal_event_spiking.def
```

Do not claim this succeeds until P0-11 records the actual build. Revisit the container only when the
DVS-Lip loader's verified dependencies are known.

## Interactive validation

```bash
singularity shell --nv \
  --bind /path/to/repository:/workspace \
  --bind /home/users/<username>:/local \
  /home/users/<username>/temporal-event-spiking.sif

cd /workspace
python -m pip install -e . --no-deps
pytest -q
ruff check src tests
python -m compileall -q src tests
bash -n scripts/*.sh
git diff --check
```

Record Python, Torch, CUDA, GPU and relevant package versions in the run artifact. The current runner
does not yet produce the charter's full `environment.json`; this is an open P0/P1 defect.

## Long runs

Create output roots on local server storage, not the network home, and export:

```bash
export ETSR_ARTIFACT_ROOT=/local/etsr/artifacts
export ETSR_CHECKPOINT_ROOT=/local/etsr/checkpoints
screen -S <descriptive-run-name>
```

Launch only a documented command/config. Detach with `Ctrl-a d`, verify with `screen -ls`, and
re-enter with `screen -r <name>`. Push code/documentation periodically, but never raw datasets,
checkpoints, caches, writable sandboxes or credentials.

## Historical DVS-GC commands

Legacy commands remain documented in [`archive/dvsgc/smilies_setup.md`](archive/dvsgc/smilies_setup.md).
They may be used for P0-10 regression only, not to reopen DVS-GC research. There is no DVS-Lip launch
command yet; inventing one before loader/config implementation would be misleading.

## End-of-run checklist

- confirm process exit status and complete log;
- verify config, manifests, environment, metrics and checkpoint paths;
- copy compact persistent artifacts to approved storage;
- record the run in `EXPERIMENT_LEDGER.md` if it changes a decision;
- free unnecessary local checkpoints only after retention is confirmed.
