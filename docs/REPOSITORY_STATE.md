# Repository state

**Observed:** 2026-08-21
**Scope:** facts verified from Git, source, scripts, configs and local commands
**Implementation changes in the current iteration:** bounded P0 smoke, state-isolation regressions,
environment capture and frozen-code annotations/policy

## Git transition

- **FACT:** source snapshot is `806c0aa62b1aff577fa1158a1a6302866f4271f3`.
- **FACT:** `master`/`origin/master` point to `02cd224` at inspection time.
- **FACT:** `developer` was created locally from `806c0aa`.
- **FACT:** annotated tag `dvsgc-audit-complete-2026` points to `806c0aa`.
- **FACT:** the only configured remote is `origin` at
  `https://github.com/Bernuz2003/Temporal-Events-Spiking.git`; no SMILIES GitLab remote is configured.
- **FACT:** a live `git ls-remote` query found `master` at `02cd224` and
  `refactor/mechanistic-temporal-audit` at `806c0aa`, but no remote `developer` branch.
- **FACT:** no push, remote addition or remote modification was performed.
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

The tree contains 35 pytest test functions across:

- model shape/backprop and trace/prefix equivalence;
- grouping, split embargo and exact reverse pairs;
- perturbations and count-preserving redistribution;
- classification/content/order/prefix/aggregation metrics;
- causal alignment and activation patching;
- operation and Horowitz calculations.
- bounded smoke validation/orchestration, environment serialization and LIF state isolation.

Current gaps relevant to the new phase:

- no DVS-Lip/raw-event contract tests;
- no stateful streaming step/chunk/reset tests;
- no recurrent state/memory profiler tests;
- the new smoke/state/provenance tests have not yet run in a dependency-complete environment.

## Known drift and defects

| Item | Evidence | Disposition |
|---|---|---|
| stale `make smoke` target and deleted files | Makefile/history at transition | bounded config, CLI and launcher restored; runtime pending |
| previous repository tree listed deleted smoke files | archived `repository_tree.md` | active state now generated from actual tree |
| previous validation document reported obsolete/incomplete suite state | archived `validation.md` | replaced by this evidence record |
| notebook lifecycle was undefined | untracked 1.46 MB notebook | archived and indexed by D007 |
| artifact schema lacks full hardware state profile and recipe manifest | runner vs charter §40 | environment capture fixed; remaining work P1/P4 |
| official-test flag is inferred from dataset name in one legacy training path | `runner.py` | must not be reused for DVS-Lip embargo |

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
| pytest/smoke execution | awaiting supported runtime; not claimed |

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
