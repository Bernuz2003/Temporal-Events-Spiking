# Frozen DVS-GC code boundary

**Status:** policy active; physical relocation/removal deferred until runtime regression passes

## Purpose

The closeout tag preserves the complete DVS-GC phase, but the working tree still needs parts of that
code for P0 regression. This document prevents those frame-first assumptions from becoming hidden
dependencies of DVS-Lip while avoiding an untested large move.

## Frozen paths

The following are historical/regression surfaces:

```text
configs/mechanistic_audit_dvsgc_order2.yaml
configs/temporal_audit_dvsgc_chain4.yaml
configs/temporal_audit_dvsgc_order2.yaml
scripts/legacy/dvsgc/prepare_dvsgc.sh
scripts/legacy/dvsgc/prepare_matched_dvsgc.sh
scripts/legacy/dvsgc/run_mechanistic_audit.sh
scripts/legacy/dvsgc/train_audit_seed.sh
scripts/legacy/dvsgc/train_temporal_baseline.sh
scripts/legacy/dvsgc/train_temporal_baseline_screen.sh
src/etsr/data/dvsgc.py
src/etsr/data/matched_dvsgc.py
src/etsr/evaluation/mechanistic.py
```

The related CLI commands are `temporal-audit`, `prepare-matched-dvsgc` and `mechanistic-audit`.

Allowed changes:

- a failing regression or security/correctness fix with tests;
- extraction of a genuinely dataset-neutral utility into an active module;
- path/import repair caused by an approved reorganization;
- explicit removal after the retirement gate below.

Not allowed:

- new DVS-GC scientific metrics or architecture experiments;
- using matched DVS-GC frames as the DVS-Lip sample contract;
- adding DVS-Lip branches inside the frozen mechanistic orchestrator;
- silently changing historical protocol semantics.

## Active reusable surfaces

The following remain candidates for reuse, subject to normal review:

```text
src/etsr/config.py
src/etsr/reproducibility.py
src/etsr/training/
src/etsr/profiling/
src/etsr/utils/
src/etsr/models/mini_qkformer.py       # baseline/candidate spatial model only
src/etsr/models/spiking.py             # current LIF baseline only
selected generic evaluation metrics
```

Reuse does not freeze their current APIs. In particular, the dense `(frames, targets, indices)`
training contract must be replaced or generalized for raw-event samples.

## Dependency rule for DVS-Lip

New active DVS-Lip modules must not import from:

```text
etsr.data.dvsgc
etsr.data.matched_dvsgc
etsr.evaluation.mechanistic
```

If a needed utility exists only there, extract it under a neutral name, add independent tests and
leave the frozen caller using the extracted function. Do not import the legacy module for
convenience.

## Why code is not removed yet

- the full suite and bounded smoke now pass in the owner's supported environment;
- P0-10 still requires one historical sanity benchmark;
- existing configs/scripts provide the command surface for that check;
- removing the legacy command surface before that benchmark would prevent the required regression.

## Retirement gate

Physical relocation or deletion may proceed after:

1. full pytest passes in the supported environment;
2. the bounded smoke passes;
3. P0-10 records the required historical sanity result or the owner explicitly waives it;
4. no active DVS-Lip module imports a frozen path;
5. the closeout tag and remote retention policy are verified;
6. `REPOSITORY_TREE.md`, CLI help and archived instructions are updated in the same change.

The tag makes removal recoverable; it does not make untested removal responsible.
