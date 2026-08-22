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

## D009 — Permit primary-source review while external P0 gates remain open

- **Date:** 2026-08-21
- **Question:** must all work stop while historical data, SMILIES access and the GitLab endpoint are
  unavailable?
- **Evidence:** local P0 hardening is verified by 35 passing tests and a clean, bounded end-to-end
  smoke at `0b7c552`; remaining P0-10/P0-11/P0-13 require external data, infrastructure or owner
  information. Source review changes no dataset/model/runtime contract.
- **Decision:** start P1-01 literature and official-code review in parallel, limited to evidence and
  protocol documentation. Do not implement the DVS-Lip loader, model or training recipe until the
  P0 gate and relevant source record are complete.
- **Rejected:** idle while safe uncertainty-reduction work is available; silently treating external
  blockers as waived; implementing against an unverified dataset protocol.
- **Reversal condition:** a reviewed source conflicts with the charter or reveals a licensing/data
  condition that requires owner direction; pause and record the conflict.

## D010 — Review sources just in time and resume implementation order

- **Date:** 2026-08-21
- **Question:** complete the whole literature register before writing further code, or review each
  source when the active implementation task requires it?
- **Evidence:** S001/S003/S006 already establish the dataset/protocol constraints needed for the
  current gate. Reviewing unrelated future mechanisms now would separate evidence from the design
  decision it must constrain and postpone executable progress. The scientific owner explicitly
  requested implementation order with source review on demand.
- **Decision:** preserve completed reviews, pause broad P1-01/P1-02/P1-03 expansion, and consult the
  next primary source immediately before its mechanism/comparison becomes active. Implement only
  the train-only DVS-Lip preflight needed to resolve P1-04; do not implement the loader, encoder,
  temporal core or training path while their gates remain open.
- **Rejected:** front-load every listed paper; discard completed source records; implement an
  external mechanism without first reviewing its primary paper and official code.
- **Reversal condition:** a near-term architecture decision requires multiple sources to be reviewed
  together to avoid an invalid novelty or comparability conclusion.
