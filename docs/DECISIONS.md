# Decision log

Append-only. A later entry may supersede an earlier decision; historical entries are not rewritten.

## D001 — Continue the existing repository

- **Date:** 2026-08-21
- **Question:** rewrite, return to `master`, or continue from the audited refactor?
- **Evidence:** `806c0aa` contains reusable split/provenance, tracing, profiling and validation
  infrastructure and is a descendant of `master` at `02cd224`.
- **Decision:** continue from `refactor/mechanistic-temporal-audit` at `806c0aa` on `developer`.
- **Why:** preserves tested infrastructure and research provenance while allowing targeted new APIs.
- **Rejected:** clean rewrite; return to `master`; direct development on `master`.
- **Reversal condition:** repository-level defects make targeted refactoring more costly or less
  verifiable than a migration, supported by an explicit review.
- **Refs:** tag `dvsgc-audit-complete-2026`, branch `developer`.

## D002 — Close DVS-Gesture-Chain as an active research phase

- **Date:** 2026-08-21
- **Question:** continue the mechanistic audit or move to DVS-Lip?
- **Evidence:** the charter accepts the audit's minimal conclusion and identifies DVS-GC order-2 as
  too shortcut-solvable for the new objective.
- **Decision:** freeze DVS-GC as historical evidence and regression material.
- **Rejected:** further metrics, probes or architecture search on DVS-GC.
- **Reversal condition:** explicit reopening by the scientific owner for a named scientific reason.
- **Refs:** `docs/archive/dvsgc/`, `dvsgc-audit-complete-2026`.

## D003 — Revise the role of Mini-QKFormer

- **Date:** 2026-08-21
- **Question:** retain Mini-QKFormer as the assumed full final backbone?
- **Evidence:** current code couples QKTA/SSA with LIF and temporal mean; charter evidence indicates
  that temporal mechanism choice is material on DVS-Lip.
- **Decision:** treat Mini-QKFormer as a baseline/candidate spatial backbone and temporal memory as an
  explicit variable.
- **Rejected:** discard QKTA without an ablation; preserve it by attachment; copy a large
  bidirectional GRU.
- **Reversal condition:** controlled architecture experiments show current LIF dynamics are
  competitive and explicit state gives no Pareto benefit.

## D004 — Stabilize training before representation conclusions

- **Date:** 2026-08-21
- **Question:** may the representation bake-off start with an unstable recipe?
- **Decision:** no. Tune E0 under a bounded budget, freeze a recipe ID, then compare representation
  and core.
- **Why:** training changes can otherwise mask or imitate architecture effects.
- **Reversal condition:** none without a replacement confound-control protocol approved by the owner.

## D005 — Treat 500k as an aggressive target, not a proven feasible cap

- **Date:** 2026-08-21
- **Question:** assume that ~500k can attain the desired DVS-Lip accuracy?
- **Decision:** retain ~500k as preferred target and run aligned 0.5M/1M/2M pilots before enforcing
  it as a final cap.
- **Rejected:** silent model growth; declaring infeasibility from literature at different scales.
- **Reversal condition:** capacity evidence justifies a revised cap and the owner approves it.

## D006 — Separate stable charter, mutable registers and frozen archive

- **Date:** 2026-08-21
- **Question:** keep one 3,000-line document as both constitution and daily log?
- **Evidence:** the charter itself prescribes child documents; legacy active docs contradicted the
  current tree and phase.
- **Decision:** preserve the full charter as `PROJECT_CHARTER.md`, maintain small operational child
  documents, and move the DVS-GC narrative to `docs/archive/dvsgc/`.
- **Rejected:** duplicate authoritative charters; delete legacy documents; rewrite historical claims
  as if they described DVS-Lip.
- **Reversal condition:** the structure demonstrably causes duplicate state; consolidate through a
  new decision while retaining history.

## D007 — Preserve the DVS-GC analysis notebook as a frozen record

- **Date:** 2026-08-21
- **Question:** delete, ignore, or version the previously untracked notebook?
- **Evidence:** it contains 43 cells, 39 outputs and DVS-GC behavioral/mechanistic analysis; it is
  part of the closed phase rather than the active DVS-Lip workflow.
- **Decision:** retain it unchanged under `notebooks/archive/dvsgc/`, index it, record its checksum
  and prohibit use as an authoritative active protocol.
- **Rejected:** discard outputs; present it as a current pipeline; leave lifecycle ambiguous.
- **Reversal condition:** extract a reproducible report/artifact and supersede it without destroying
  the archived source.

## D008 — Restore a bounded smoke workflow

- **Date:** 2026-08-21
- **Question:** remove the stale `make smoke` target or restore executable end-to-end validation?
- **Evidence:** Git history contains a one-epoch synthetic config and launcher; the current training,
  profiling and behavioral-audit paths still need a dataset-independent integration check.
- **Decision:** restore the workflow with a dedicated `etsr smoke` command and reject configs that
  exceed strict dataset/epoch/worker/profiling bounds.
- **Why:** removing the target repairs documentation drift but leaves the P0 smoke gate untested;
  bounded validation prevents accidentally launching a real or long experiment.
- **Rejected:** silently reusing an arbitrary training config; embedding unvalidated Python in the
  shell launcher; running the smoke in the dependency-incomplete local environment.
- **Reversal condition:** replace it with a faster integration harness that exercises the same
  training/checkpoint/holdout/profile/audit artifact path.
