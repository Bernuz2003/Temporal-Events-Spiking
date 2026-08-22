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

## D011 — Use an explicit sample-level development split when speaker metadata is unavailable

- **Date:** 2026-08-22
- **Owner:** scientific owner, implemented in repository policy by the project agent.
- **Question:** must missing dataset terms and sample-to-speaker metadata block DVS-Lip development
  indefinitely when the archive came from the official project download and contains only `train/`
  and `test/`?
- **Evidence:** the official project page links its dataset through Google Drive and publishes no
  dataset license or speaker mapping. The owner obtained that ZIP from the official link, observed
  only `train/` and `test/`, and independently found no terms or mapping. The real train-only
  preflight verified 100 classes, 14,896 samples and the expected structured event layout without
  touching `test/`. Integer filenames do not encode a verified speaker identity.
- **Decision:** absence of speaker mapping and dataset-specific terms is no longer a protocol
  blocker. Record the official provenance and absence explicitly; proceed only for local thesis
  research without claiming a dataset license or redistribution right. Build development train and
  validation from official `train/` using the versioned 80/20, per-class, deterministic hash-ranked
  sample policy with seed `314159`. Every manifest and result must declare
  `speaker_identity_available=false`, `speaker_disjoint=false` and that validation is not an
  unseen-speaker estimate. The official `test/` remains embargoed and retains the published
  speaker-disjoint evaluation role.
- **Rationale:** 80/20 preserves the charter's intended 24/6 development proportion; class
  stratification avoids vocabulary imbalance; hash ranking makes assignments reproducible without
  depending on a pseudorandom-library version. The seed has no scientific interpretation.
- **Still required:** full official-train content hashing, physical test quarantine, code-level
  embargo and explicit caveats in every comparison. These requirements are not waived.
- **Rejected:** infer speaker identity from filenames or ordering; use `test/` for validation;
  describe the fallback as speaker-disjoint; treat repository code licenses or the download link as
  a dataset license grant; version or redistribute the dataset.
- **Reversal condition:** an authoritative sample-to-speaker mapping appears before recipe freeze.
  In that case create and validate a speaker-disjoint development split, record a superseding
  decision and invalidate incompatible sample-level manifests before further model selection.

## D012 — Isolate DVS-Lip without relocating the frozen audit pipeline

- **Date:** 2026-08-22
- **Decision:** retain the working DVS-GC data, runner, model and temporal/mechanistic audit modules
  in place as frozen regression code. Put all new phase-specific implementation under
  `src/etsr/dvslip/`, and use command-local imports so DVS-Lip commands do not load legacy audit
  orchestration.
- **Why:** deleting loses cheap regression/provenance; moving the interconnected legacy graph only
  creates import churn. A vertical DVS-Lip package provides the required boundary with minimal code.
- **Rule:** DVS-Lip may use genuinely compatible shared utilities, but must not enter the old
  frame-first dataset factory or add compatibility layers for it.
- **Reversal condition:** move or remove frozen modules only if a concrete active dependency blocks
  DVS-Lip implementation or their maintenance cost becomes measurable.

## D013 — Replace physical test quarantine with a logical embargo

- **Date:** 2026-08-22
- **Owner:** scientific owner, on recommendation of the project agent.
- **Decision:** the official `test/` directory may remain beside `train/`. Development APIs accept
  only the official `train/` root and the `train`/`validation` development assignments; they neither
  discover nor load the test sibling. Official-test evaluation will be added only after the model,
  representation and recipe freeze. This supersedes the physical-quarantine requirement in D011
  and the earlier reference documents, while preserving their prohibition on test-based selection.
- **Why:** for this single-owner local thesis project, moving the directory adds operational work
  without strengthening the scientific rule that matters. The former preflight blocker was also
  unconditional and therefore could not verify quarantine. Explicit code paths and final-only test
  evaluation are the smallest sufficient protection.
- **Reversal condition:** require stronger filesystem isolation if development becomes automated or
  multi-user, or if an active code path can reach the official test before freeze.

## D014 — Fix the E0 physical-time count representation

- **Date:** 2026-08-22
- **Evidence:** all 14,896 train samples are valid. Duration is 0.829/1.079/1.358 s at
  p05/median/p95 and reaches 1.889 s; the median event rate is 8.25k events/s. Train and validation
  distributions are closely aligned. Published baselines instead use sample-normalized voxels or a
  1.2 s cut, neither of which preserves the physical-time contract selected for this project.
- **Decision:** E0 uses the full 128×128 sensor, separate OFF/ON channels, a fixed 2,000,000 us
  window, 50,000 us bins (`T=40`) and exact unsigned 8-bit counts. Shorter samples are zero-padded in
  physical time. Timestamps outside the window or voxel counts above 255 raise an error; neither is
  silently clipped. The encoder is stateless and reports zero persistent state.
- **Why:** 2.0 s covers the observed train maximum with explicit headroom. A 50 ms bin gives about
  22 active steps at median duration and preserves a clear higher-rate E2 comparison without
  copying a literature timestep convention. Uint8 is the simplest low-bit exact baseline.
- **Rejected:** duration normalization, the published 1.2 s cut, signed polarity cancellation,
  implicit float frame truth and silent saturation.
- **Reversal condition:** revise before training if the full-train encoding check finds uint8
  overflow, or if the resulting memory/runtime makes bounded E0 training infeasible. The official
  test cannot be inspected to tune this choice.

## D015 — Start recipe stabilization with one minimal, evidence-backed E0 candidate

- **Date:** 2026-08-22
- **Evidence:** exhaustive E0 encoding preserves every event in all 14,896 development samples; the
  observed maximum voxel count is 17, safely below the fixed cap of 255. S001 and S003 use learning
  rates around `3e-4`; the targeted S005 code review at commit `519fd2e` found AdamW at `1e-3`
  with total batch 256 (four devices × 64), four warm-up epochs, cosine decay, label smoothing and
  clipping. Those models and input semantics are not directly transferable.
- **Decision:** begin with candidate `dvslip_e0_r0`: the 500,708-parameter Mini-QKFormer, E0,
  AdamW at `3e-4`, weight decay `5e-4`, four-epoch linear warm-up from 1%, cosine decay to `1e-6`,
  label smoothing 0.1, clipping at 1.0 and 64 epochs. Physical batch 4 plus eight-step gradient
  accumulation gives effective batch 32 without requiring a large activation footprint.
- **Augmentation:** train-only horizontal flip with probability 0.5. It changes spatial geometry but
  neither physical timing nor event count. Validation is deterministic. Temporal masking,
  resize/shear, erasing and event-rate perturbation are excluded until evidence shows they are
  needed; adding them now would enlarge the search and weaken E0 comparability.
- **Budget:** run `r0` with seed 42. At most one fallback configuration may be run, changing only the
  learning rate to `1e-3`, and only if `r0` is finite but shows inadequate optimization. A memory
  failure is handled by changing micro-batch and inverse accumulation together so effective batch
  remains 32; it is not a new scientific candidate. Maximum stabilization budget: two 64-epoch
  runs, one seed each, no official-test access.
- **Freeze gate:** inspect loss/accuracy curves, validation Macro-F1, best epoch and runtime before
  freezing. A crash, non-finite values or absence of clear learning cannot produce a frozen recipe.
  Expanding the budget or changing another recipe dimension requires owner approval.

## D016 — Measure the physical-duration shortcut before running E0

- **Date:** 2026-08-22
- **Evidence:** an independent full scan of all 14,896 official-train samples gives class-explained
  variance η²=0.279 for duration, 0.104 for event count, 0.080 for ON fraction and 0.275 for the
  number of occupied 50 ms bins. Occupied-bin count and duration have Pearson correlation 0.996;
  E0 therefore makes a class-informative duration proxy observable through zero padding. The
  temporal mean and flattened-time BatchNorm can interact with this proxy, but actual shortcut use
  by the model is not established before measurement.
- **Decision:** retain D014's physical-time E0 because physical duration is part of the target
  streaming contract, but make a fixed global-statistic control a pre-r0 gate. Report both raw
  duration metadata and the E0-observable occupied-bin counterpart, using only development
  train/validation. During r0, save per-sample occupied bins, duration, event count, polarity,
  correctness and logit margin so correlations can be inspected after training.
- **Comparability caveat:** literature alignment is source-specific. MSTP and NSA normalize sample
  duration, while the reviewed SpikGRU path uses a fixed physical window with truncation/padding and
  may expose a related cue. Do not claim that every published anchor removes duration, and do not
  interpret a numerical advantage as architecture-only evidence.
- **Rejected now:** normalize timestamps and silently abandon the physical-time objective; change to
  masked/last/gated readout before measuring the declared mean baseline; treat η² alone as model
  accuracy or causal shortcut use.
- **Reversal condition:** if the E0-observable control is non-trivial or r0 correctness/margin varies
  materially with occupied-bin count, prioritize the already planned P2 readout/shortcut controls.
  Otherwise retain the caveat but do not expand the shortcut study.
