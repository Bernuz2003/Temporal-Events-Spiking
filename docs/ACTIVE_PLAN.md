# Active plan

**Updated:** 2026-08-21
**Current phase:** P0 — reproducibility and repository transition
**Current iteration:** P1-04 train-only DVS-Lip preflight under the remaining external gates
**Current iteration status:** preflight implementation verified; external data evidence required
**Overall P0 status:** active, blocked on P0-10/P0-11/P0-13

## Scientific question

Can the repository verify the real DVS-Lip training archive and its protocol inputs without touching
the official test or forcing raw events through the DVS-GC frame factory?

## Current evidence

- **FACT:** `developer` and `origin/developer` point to `0b7c552`; the current working tree contains
  the preflight iteration, while the frozen DVS-GC snapshot remains `806c0aa`.
- **FACT:** the current data/training path consumes dense frame tensors and fixed tuple batches.
- **FACT:** `MultiStepLIF` resets local membrane state on every sequence forward, but the project has
  no generic streaming-state API.
- **FACT:** profiling omits recurrent state traffic, buffers, gates and persistent-state precision.
- **FACT:** active README/tree/validation/smoke references had drifted from the file tree.
- **FACT:** the owner-run Python 3.10 environment passed all 35 tests and the bounded CPU smoke.
- **FACT:** smoke provenance, checkpointing, holdout, profiling and audit artifacts are present and
  internally consistent; its one-batch firing profile is all zero.
- **FACT:** the only configured remote is GitHub `origin`, while the charter names a SMILIES GitLab
  workflow; `origin/developer` now exists, but no intended GitLab remote is inferable.
- **FACT:** S001, S003 and S006 expose the published DVS-Lip layout but no sample-to-speaker map;
  each reviewed training path consults the official test during model development.
- **FACT:** NSA's ALR comparisons are approximately 9.5M-parameter experiments; they demonstrate
  temporal-mechanism sensitivity, not preferred-target-scale feasibility.
- **FACT:** `preflight-dvslip` accepts only an official `train/` root, validates exact layout,
  representative structured samples, explicit speaker/split/terms metadata and optional full hashes.
- **FACT:** the semantic 100-class/25-pair Acc1/Acc2 manifest is versioned and unit tested.
- **INFERENCE:** preserving the repository is valuable, but DVS-Lip must not be forced into the
  frame-first factory/training contract.

## Iteration hypothesis

**HYPOTHESIS:** a strict train-only preflight can convert the missing external DVS-Lip evidence into
explicit machine-readable blockers, so the raw loader can later be implemented from inspected facts
without exposing the official test or guessing speaker identity.

## Work in this iteration

| Work item | Status | Evidence |
|---|---|---|
| restore a bounded generic smoke workflow | DONE | owner-run smoke passed; required artifacts verified |
| add explicit LIF reset/isolation regressions | DONE | two tests included in 35-test passing suite |
| write `environment.json` for every training run | DONE | artifact/hash/resolved provenance verified |
| define frozen DVS-GC code boundary | DONE | policy, annotations and regressions verified |
| run static checks | DONE | Ruff, compileall, shell syntax and diff check pass |
| run pytest and end-to-end smoke | DONE | `35 passed`; smoke status `passed` at `0b7c552` |
| review primary DVS-Lip protocol sources needed by this gate | DONE | S001/S003/S006 exact versions recorded |
| implement train-only DVS-Lip preflight | DONE | CLI, Make target and strict validation module |
| version and validate paper-semantic Acc1/Acc2 groups | DONE | 100 classes, 25 pairs, 50/50 groups |
| add synthetic preflight regressions | DONE | 6 tests; full suite 41 passed |
| run preflight on the real official-train archive | BLOCKED | archive path and external metadata unavailable |

## P0 go/no-go

Current decision: **NO-GO for the raw loader/model/training path while P0-10/P0-11/P0-13 and the
P1-04 data gate remain open; GO for the bounded train-only preflight under D010.**

The locally executable P0 hardening is closed with runtime evidence. P0 itself remains open until
the historical sanity, SMILIES/CUDA verification and GitLab integration tasks in
[`TASKS.md`](TASKS.md) are completed or explicitly waived. The smoke proves plumbing only: chance
accuracy and zero profiled firing do not validate learning or scientific adequacy.

## Inputs

- [`PROJECT_CHARTER.md`](PROJECT_CHARTER.md), revision v2;
- repository at `806c0aa` plus the untracked charter/notebook supplied for transition;
- source, tests, configs, shell scripts and Git history inspected on 2026-08-21;
- frozen DVS-GC documents in [`archive/dvsgc/`](archive/dvsgc/README.md).

## Outputs

- safe `etsr smoke` entrypoint, `configs/smoke.yaml` and launcher;
- LIF state-isolation tests;
- per-run `environment.json` plus provenance test coverage;
- frozen-code boundary document and CLI/source annotations;
- exact runtime commands for the scientific owner;
- `preflight-dvslip`, its machine-readable report and paper-semantic class manifest.

## Blocked by

- project data/checkpoints for a historical sanity run;
- SMILIES/Singularity/CUDA access for server verification.
- confirmed GitLab remote URL/permissions before any integration push.
- DVS-Lip official-train path, authoritative sample-to-speaker CSV and dataset terms.
- a documented physical quarantine arrangement for the official DVS-Lip test partition.

## Next decision

Run the quick preflight on the actual official-train path, then supply the authoritative speaker map
and dataset terms. Do not infer a 24/6 split from integer filenames. Broad literature review remains
paused under D010 and resumes only when the active implementation needs a source. The historical
DVS-GC artifact/data, SMILIES access and intended GitLab endpoint also remain owner inputs.

## Verified runtime evidence

- command: `make test`; result: 35 passed in 3.08 s;
- command: `make smoke`; run: `smoke_synthetic__20260821_193155__seed7`, status `passed`;
- commit/dirty state: `0b7c552`, clean;
- runtime: Python 3.10.20, Torch 2.12.1+cu130, CPU selected, CUDA unavailable;
- environment SHA-256:
  `8e024dd862d253affa0ec88071f3426d72fefc38a8b0ca03a90ca8c0ce33f4cc`;
- test embargo: `official_test_used: false`.

Current implementation verification:

- `ruff check src tests`: pass;
- `python -m compileall -q src tests`: pass;
- `bash -n scripts/*.sh`: pass;
- `python -m pytest -q`: 41 passed in 4.36 s using the supported external project environment;
- no training or real-dataset command was executed.
