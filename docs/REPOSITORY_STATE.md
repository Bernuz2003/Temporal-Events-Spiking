# Repository state

**Observed:** 2026-08-21
**Scope:** facts verified from Git, source, scripts, configs and local commands
**Implementation changes through `0b7c552`:** bounded P0 smoke, state-isolation regressions,
environment capture and frozen-code annotations/policy
**Current uncommitted iteration:** DVS-Lip train-only preflight, class manifest and regressions

## Git transition

- **FACT:** source snapshot is `806c0aa62b1aff577fa1158a1a6302866f4271f3`.
- **FACT:** `master`/`origin/master` point to `02cd224` at inspection time.
- **FACT:** `developer` was created from `806c0aa`; local and `origin/developer` point to
  `0b7c5528dc54476650f192afdfbf38ddb3326734`. The working tree contains the current documented
  preflight implementation.
- **FACT:** annotated tag `dvsgc-audit-complete-2026` points to `806c0aa`.
- **FACT:** the only configured remote is `origin` at
  `https://github.com/Bernuz2003/Temporal-Events-Spiking.git`; no SMILIES GitLab remote is configured.
- **FACT:** the earlier live query found no remote `developer`; after owner commit/push,
  `origin/developer` is present at `0b7c552`.
- **FACT:** no GitLab remote, remote addition or remote modification was performed by the agent.
- **FACT:** the supplied charter and notebook were untracked before this refactor.

## Implemented architecture and contracts

### Data

- `IndexedDataset` canonicalizes samples to dense float frames `[T,C,H,W]`.
- `DatasetBundle` exposes `train`, `validation`, `holdout`, but semantic holdout enforcement remains
  protocol-specific.
- The factory supports `dvsgc`, `matched_dvsgc` and `synthetic_temporal_order`; it has no DVS-Lip
  raw-event loader.
- `matched_dvsgc` builds grouped train/checkpoint-validation/development-audit splits from official
  DVS-GC training events and checks manifest embargo flags.
- No DVS-Lip files or metadata are present under local `data/`. Primary-source review shows that the
  three public loaders consume `train|test/<word>/<integer>.npy` but expose no sample speaker ID.
- A separate `dvslip_preflight` module now validates only a supplied official `train/` root. It is
  not registered in the dense-frame factory and cannot open an official `test/` root.

### Training and model

- The engine loops over `(frames, targets, indices)` and calls `model(frames)`; representation and
  model input are coupled.
- Mini-QKFormer consumes `[B,T,C,H,W]`, permutes time first internally, applies hierarchical
  embeddings, QK token gating, SSA and temporal mean before a linear head.
- Each `MultiStepLIF.forward` initializes membrane state from zeros and propagates it only within that
  call. This design prevents persistent cross-call state by construction, but no explicit
  step/chunk/reset API exists.
- `forward_with_trace` returns time-resolved features/logits; tests compare its final output and every
  cumulative prefix with explicit forwards.
- Every `train_experiment` now writes an allow-listed `environment.json`, stores its SHA-256 in the
  resolved config and exposes its path/hash in `summary.json`.

### Evaluation and profiling

- Behavioral audit includes temporal perturbations, paired prediction analysis and prefix metrics.
- Mechanistic audit includes grouped splits, content/order factorization, inverse consistency,
  duration redistribution, probes, shortcut baselines and activation patching.
- Operation profiling covers tagged Conv/Linear MAC/AC and custom attention mixing using observed
  activity.
- Horowitz output is explicitly labeled an arithmetic proxy and excludes memory/data movement.
- There is no generic accounting for persistent state bits, reads/writes, recurrent gates, LUTs,
  circular buffers, decay updates or BRAM mapping.

## Test inventory

The tree contains 41 pytest test functions across:

- model shape/backprop and trace/prefix equivalence;
- grouping, split embargo and exact reverse pairs;
- perturbations and count-preserving redistribution;
- classification/content/order/prefix/aggregation metrics;
- causal alignment and activation patching;
- operation and Horowitz calculations;
- bounded smoke validation/orchestration, environment serialization and LIF state isolation;
- DVS-Lip train-only layout/sample inspection, external manifest coverage, split embargo and
  paper-semantic Acc1/Acc2 class groups.

Current gaps relevant to the new phase:

- no frozen DVS-Lip raw-event loader contract or dataset implementation;
- no stateful streaming step/chunk/reset tests;
- no recurrent state/memory profiler tests;
- no real-archive run or authoritative sample-to-speaker manifest to exercise those validators.

## Known drift and defects

| Item | Evidence | Disposition |
|---|---|---|
| stale `make smoke` target and deleted files | Makefile/history at transition | bounded config, CLI and launcher restored; runtime passed |
| previous repository tree listed deleted smoke files | archived `repository_tree.md` | active state now generated from actual tree |
| previous validation document reported obsolete/incomplete suite state | archived `validation.md` | replaced by this evidence record |
| notebook lifecycle was undefined | untracked 1.46 MB notebook | archived and indexed by D007 |
| artifact schema lacks full hardware state profile and recipe manifest | runner vs charter §40 | environment capture fixed; remaining work P1/P4 |
| official-test flag is inferred from dataset name in one legacy training path | `runner.py` | must not be reused for DVS-Lip embargo |
| official DVS-Lip repositories use `test/` for per-epoch checkpoint selection | S001/S003/S006 exact code commits | reject training loops; require quarantined test and train-speaker validation |
| public DVS-Lip sources omit sample-to-speaker mapping and dataset terms | paper/supplement/repositories/download page | P1-04 blocked; see `DVSLIP_PROTOCOL.md` |
| MSTP public Acc1/Acc2 helper appears reversed | S001 supplement class list vs official helper | use semantic group names and tested paper mapping |
| DVS-Lip archive evidence is unavailable locally | `data/` plus preflight gate | run `preflight-dvslip` when owner supplies train root and metadata |

## Local verification baseline

Environment observed:

```text
Python 3.14.6 (outside package constraint >=3.10,<3.13)
ruff 0.15.18
torch: unavailable
numpy: unavailable
PyYAML: unavailable
pytest: unavailable
Singularity/Apptainer: unavailable
```

Commands and outcome before documentation edits:

| Command | Outcome | Meaning |
|---|---|---|
| `pytest -q` | BLOCKED: command unavailable | no runtime test claim |
| `ruff check src tests` | PASS | static lint under installed Ruff |
| `python -m compileall -q src tests` | PASS | syntax/bytecode compilation only |
| `bash -n scripts/*.sh` | PASS | shell syntax only |
| `git diff --check` | PASS | pre-refactor diff whitespace check |

Final documentation-specific checks:

| Check | Outcome |
|---|---|
| required transition artifacts | 18/18 present |
| local Markdown links | 34 files checked, 0 missing |
| frozen legacy preservation | 12/12 files byte-identical to `HEAD:docs/*` |
| notebook checksum | matches the value in `notebooks/README.md` |
| `ruff check src tests` | PASS |
| `python -m compileall -q src tests` | PASS (syntax only) |
| `bash -n scripts/*.sh` | PASS |
| `git diff --check`, including intent-to-add new files | PASS |
| implementation/config/script diff | none |
| `python -m pytest -q` | BLOCKED: `No module named pytest` |

Current P0 implementation static checks:

| Check | Outcome |
|---|---|
| `ruff check src tests` | PASS |
| `python -m compileall -q src tests` | PASS |
| `bash -n scripts/*.sh` | PASS |
| `make -n test`, `make -n lint`, `make -n smoke` | PASS; commands resolve without execution |
| `git diff --check` | PASS |
| owner `make test` | PASS: 35 tests in 3.08 s at clean `0b7c552` |
| owner `make smoke` | PASS: `smoke_synthetic__20260821_193155__seed7` on CPU |
| current `python -m pytest -q` | PASS: 41 tests in 4.36 s; no training invoked |
| current `ruff`/compileall/shell syntax | PASS after DVS-Lip preflight implementation |
| current Markdown local links | PASS: 37 files checked, zero missing |

Verified smoke provenance:

- Python 3.10.20 and Torch 2.12.1+cu130; CUDA compiled but unavailable, CPU selected;
- environment SHA-256
  `8e024dd862d253affa0ec88071f3426d72fefc38a8b0ca03a90ca8c0ce33f4cc`, identical in the
  environment artifact, resolved config and run summary;
- `official_test_used: false`; required checkpoint, environment, profile, summary and audit files
  exist;
- 31,460 trainable parameters, one epoch and 16 holdout samples under bounded smoke limits;
- chance accuracy 0.25/macro-F1 0.10 and zero firing in the profiled batch: integration pass, not a
  numerical or scientific validation.

The captured Python executable is located in a virtual environment under the separate historical
path `PredictiveErrorGatedSpikingTransformer`. This is faithfully recorded provenance and does not
invalidate the integration path, but the environment must not be described as an isolated
repository-local environment. SMILIES/container verification remains required.

## Interpretation

- **INFERENCE:** the repository is worth continuing because the closed phase produced substantial
  reusable validation/provenance code.
- **INFERENCE:** the new phase requires targeted interface refactoring; adding another frame dataset
  to the current factory would preserve the wrong canonical abstraction.
- **OPEN QUESTION:** whether current Mini-QKFormer capacity/training is adequate for DVS-Lip cannot be
  answered from this repository state.
- **OPEN QUESTION:** actual Singularity/CUDA behavior and historical baseline plausibility require
  the intended server environment and data.
- **OPEN QUESTION:** the owner must identify the intended SMILIES GitLab endpoint before the
  charter's integration/push workflow can be followed safely.
- **OPEN QUESTION:** DVS-Lip loader work requires authoritative speaker metadata and dataset-use
  terms; neither may be inferred from the public integer filenames or source-code licenses.
