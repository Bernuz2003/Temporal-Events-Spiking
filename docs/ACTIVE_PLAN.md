# Active plan

**Updated:** 2026-08-21
**Current phase:** P0 — reproducibility and repository transition
**Current iteration:** P0 implementation hardening
**Current iteration status:** implementation complete; awaiting runtime evidence
**Overall P0 status:** active, not yet at gate

## Scientific question

Can the existing repository be trusted as a reproducible base for a raw-event DVS-Lip architecture
without carrying DVS-GC assumptions into the new pipeline?

## Current evidence

- **FACT:** the active history point is `806c0aa`; the refactor branch is a descendant of `master`.
- **FACT:** the current data/training path consumes dense frame tensors and fixed tuple batches.
- **FACT:** `MultiStepLIF` resets local membrane state on every sequence forward, but the project has
  no generic streaming-state API.
- **FACT:** profiling omits recurrent state traffic, buffers, gates and persistent-state precision.
- **FACT:** active README/tree/validation/smoke references had drifted from the file tree.
- **FACT:** local Python 3.14.6 lacks runtime dependencies; only static checks can be completed here.
- **FACT:** the only configured remote is GitHub `origin`, while the charter names a SMILIES GitLab
  workflow; the live GitHub remote has no `developer` branch at verification time.
- **INFERENCE:** preserving the repository is valuable, but DVS-Lip must not be forced into the
  frame-first factory/training contract.

## Iteration hypothesis

**HYPOTHESIS:** a bounded synthetic smoke workflow, explicit LIF isolation regressions and a
machine-readable environment snapshot can close the locally implementable P0 defects without
mixing DVS-Lip architecture work into repository validation.

## Work in this iteration

| Work item | Status | Evidence |
|---|---|---|
| restore a bounded generic smoke workflow | PARTIAL | implementation/static checks complete; run pending |
| add explicit LIF reset/isolation regressions | PARTIAL | two tests added; pytest pending |
| write `environment.json` for every training run | PARTIAL | implementation/test added; runtime artifact pending |
| define frozen DVS-GC code boundary | PARTIAL | policy and annotations complete; regression pending |
| run static checks | DONE | Ruff, compileall, shell syntax and diff check pass |
| run pytest and end-to-end smoke | AWAITING USER RUNTIME | supported project environment |

## P0 go/no-go

Current decision: **NO-GO for implementation of new DVS-Lip modules until this hardening is reviewed
and its runtime checks pass.**

The documentation iteration may close when its files and links are verified. P0 itself remains open
until P0-07 through P0-14 in [`TASKS.md`](TASKS.md) are completed or explicitly waived. In
particular, static compilation is not a substitute for the unavailable pytest run.

## Inputs

- [`PROJECT_CHARTER.md`](PROJECT_CHARTER.md), revision v2;
- repository at `806c0aa` plus the untracked charter/notebook supplied for transition;
- source, tests, configs, shell scripts and Git history inspected on 2026-08-21;
- frozen DVS-GC documents in [`archive/dvsgc/`](archive/dvsgc/README.md).

## Intended outputs

- safe `etsr smoke` entrypoint, `configs/smoke.yaml` and launcher;
- LIF state-isolation tests;
- per-run `environment.json` plus provenance test coverage;
- frozen-code boundary document and CLI/source annotations;
- exact runtime commands for the scientific owner.

## Blocked by

- supported Python environment with project dependencies for pytest;
- project data/checkpoints for a historical sanity run;
- SMILIES/Singularity/CUDA access for server verification.
- confirmed GitLab remote URL/permissions before any integration push.

## Next decision

After the user returns pytest/smoke output: close or repair P0-07/P0-09/P0-14, then decide whether the
available historical data can satisfy P0-10. Do not begin the DVS-Lip loader before the P0 gate and
primary-source review are complete.

## Runtime handoff

Run from the repository root in the supported project environment, in this order:

```bash
make test
make smoke
```

Return the complete console output. A successful smoke must also create:

```text
artifacts/<SMOKE_RUN_ID>/environment.json
artifacts/<SMOKE_RUN_ID>/summary.json
artifacts/<SMOKE_RUN_ID>/profile.json
artifacts/<SMOKE_RUN_ID>/audit/audit_summary.json
artifacts/<SMOKE_RUN_ID>/smoke_summary.json
checkpoints/<SMOKE_RUN_ID>/best.pt
```

Do not run DVS-GC training or any DVS-Lip experiment for this handoff.
