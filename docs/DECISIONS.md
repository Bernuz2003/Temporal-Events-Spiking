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

## D017 — Generalize execution now, dataset contracts only from the second active benchmark

- **Date:** 2026-08-23
- **Decision:** keep training invocation config-driven and dataset-neutral, including one generic
  SMILIES launcher. Retain the current DVS-Lip vertical package through r0. When DailyDVS-200 becomes
  active, use its verified raw format together with DVS-Lip to extract the smallest neutral
  event-sample and dataset-construction interface; dataset-specific discovery, splits and protocol
  checks remain in separate adapters.
- **Why:** adding DVS-Gesture, CIFAR10-DVS or DailyDVS-200 should require a loader/config, not another
  training launcher or accumulating branches in `runner.py`. Designing the Python abstraction from
  DVS-Lip alone would nevertheless guess which metadata and split semantics are actually shared.
- **Rejected:** a registry/plugin framework for hypothetical datasets now; putting every future
  loader under `etsr.dvslip`; adding one special-case branch per dataset to the runner.
- **Reversal condition:** extract the neutral interface earlier only if a concrete pre-DailyDVS task
  needs a second raw-event dataset and provides its verified contract.

## D018 — Select the r0 physical batch from the real cost pilot

- **Date:** 2026-08-23
- **Evidence:** the clean, train-only RTX A4000 pilot at physical batch 4 and eight-step accumulation
  measured 1.24 GB peak allocated CUDA memory and 482.6 seconds of training per epoch. The complete
  64-epoch run at that throughput would spend about 8.6 hours in training alone, while most device
  memory would remain unused.
- **Decision:** run r0 with physical batch 16 and two-step accumulation, retaining the D015
  effective batch of 32. The owner approved selecting it directly from the measured headroom rather
  than spending another full epoch on a second cost pilot. Supply `CUBLAS_WORKSPACE_CONFIG=:4096:8`
  to make the configured deterministic CUDA execution effective.
- **Caveat:** inverse accumulation preserves the optimizer batch size, not exact BatchNorm
  statistics; the physical-batch change is therefore recorded as part of the final candidate
  recipe rather than described as bitwise equivalent.
- **Reversal condition:** on CUDA OOM, fall back to physical batch 8 with four-step accumulation. Do
  not change the effective batch or another scientific recipe dimension in response to memory.

## D019 — Test a wider backward surrogate before changing the forward architecture

- **Date:** 2026-08-23
- **Evidence:** `dvslip_e0_r0` remains near chance through epoch 28 with a negligible train/validation
  gap. A balanced-batch FP32 diagnostic on `last.pt` rules out a dead forward path: the final pooled
  features have mean inter-sample standard deviation 0.0486, logits vary across samples and every
  major stage fires. It instead measures mean parameter gradients 378 times smaller in stage 2
  than in the head, 62,911 times smaller in stage 1 and 1.65 million times smaller in the first
  embedding. Mean LIF-output gradient falls by about 13.2 million times from the final to the first
  LIF. Because this diagnostic is FP32, AMP is not the primary cause.
- **Decision:** retain `dvslip_e0_r0` unchanged as a failed optimization result. Candidate
  `dvslip_e0_r1` changes only the backward surrogate from the original fast-sigmoid slope 25 to the
  logistic sigmoid derivative with alpha 4; its hard-spike forward, neuron dynamics, parameters,
  representation, readout and training recipe remain unchanged. Configurations without explicit
  surrogate fields preserve the r0 behavior, and old checkpoints remain state-dict compatible.
  Before any new training, compare both backward functions on the same r0 checkpoint and balanced
  validation batch, requiring exact forward equivalence and at least a tenfold increase in mean
  gradient magnitude in both the first embedding and stage 1. Then require a small train-only
  overfit check before authorizing a full r1 run.
- **Why:** the official QKFormer implementation uses SpikingJelly LIF nodes and direct residual
  additions rather than documenting this repository's steep custom surrogate; SpikingJelly's
  default sigmoid surrogate uses alpha 4. Testing that backward-only difference is the smallest
  evidence-backed intervention. Changing residual topology, thresholds, readout or learning rate
  at the same time would destroy causal attribution.
- **Rejected now:** continue r0 to 64 epochs as if it were a viable freeze candidate; blame AMP;
  change the forward residual/LIF topology together with the surrogate; spend a full training run
  before the checkpoint and overfit gates pass.
- **Reversal condition:** do not train r1 if logits are not exactly forward-equivalent, if the
  comparison does not materially recover gradients in the first embedding/stage, or if the
  train-only overfit check still cannot learn. In that case diagnose residual topology and neuron
  dynamics one variable at a time.

## D020 — Keep one active path and make diagnostics modes, not subsystems

- **Date:** 2026-08-23
- **Owner:** scientific owner and project agent.
- **Decision:** Git history and compact result artifacts preserve failed experiments; the active
  codebase does not preserve their configuration, implementation or one-off diagnostic launcher.
  Remove the failed fast-sigmoid surrogate, the r0/r1 config duplication and the completed
  checkpoint-comparison script. Keep one canonical `dvslip_e0.yaml` with the logistic surrogate and
  configurable alpha 4. New diagnostic behavior must first be expressible as a small reusable mode
  of an existing path; create a separate subsystem only when multiple concrete tasks require it.
- **Overfit gate:** use the normal training runner on the deterministic first four official-train
  samples from each of target classes 0–15. Training and evaluation share those 64 samples;
  horizontal flip, AMP and profiling are disabled. Fifty epochs retain the recipe optimizer,
  scheduler, regularization, batch and accumulation settings, giving 100 optimizer steps. Pass only
  if the final history row has same-subset accuracy at least 95%, loss below 1.5 and finite clipping
  metrics.
- **Why:** parametrization should remove duplicated paths, not retain falsified alternatives. This
  keeps evidence reproducible while preventing each investigation from permanently expanding the
  maintenance surface.
- **Reversal condition:** retain an additional implementation or dedicated diagnostic only when it
  remains an active scientific comparator or a second verified use case cannot be served clearly
  by the shared mode.

## D021 — Align residual paths with the official QKFormer topology

- **Date:** 2026-08-23
- **Evidence:** the 100-step overfit remained at 6.25% final accuracy. Extending the same controlled
  run to 1,000 optimizer steps recovered real learning but stopped at 59.38% best and 56.25% final
  same-subset accuracy, with final loss 2.972. From epoch 150 onward every optimizer step was
  clipped. The current blocks applied a LIF after each residual sum, whereas the official QKFormer
  uses direct residual additions; its SPEDS branches also spike before, rather than after, their
  sum: <https://github.com/zhouchenlin2096/QKFormer/blob/master/imagenet/qkformer.py>.
- **Decision:** make Transformer residual additions direct and make each SPEDS branch end in its own
  LIF before direct addition. Remove the now-unused non-spiking Conv-BN helper and all post-residual
  LIF wrappers. Keep E0, model width, surrogate alpha, neuron parameters, readout, optimizer,
  scheduler, regularization, clipping and effective batch unchanged.
- **Gate:** rerun the same deterministic 16-class by 4-sample overfit for 500 epochs (1,000 optimizer
  steps). Require at least 95% final same-subset accuracy, loss below 1.5 and finite gradient
  metrics before authorizing complete E0 training.
- **Reversal condition:** if the aligned topology still fails, do not accumulate further local
  patches or launch full training. Reassess Mini-QKFormer as the active baseline and test neuron
  dynamics as a separately motivated variable.

## D022 — Use DVS-Lip to select the architecture and DVS-Gesture as the first transfer benchmark

- **Date:** 2026-08-23
- **Owner:** scientific owner and project agent.
- **Evidence:** the D021 overfit passes at 100% final same-subset accuracy and 0.894 loss. The owner
  selected DVS-Lip as the only architecture-development benchmark; other datasets compare their
  frozen baseline only with the final DVS-Lip-selected architecture. The DVS-Gesture paper defines
  11 gestures, 29 subjects and an official 23/6 subject split, with annotated intervals inside
  multi-gesture AEDAT recordings. The paper reports 1,342 instances, while maintained conversion
  libraries expose differing totals, so sample count and duration must be measured from the actual
  official archive rather than assumed.
- **Decision:** activate D017's smallest shared contract now: dataset readers return one neutral
  physical-microsecond event sample; the count encoder, encoded adapter, model and runner are shared.
  DVS-Gesture keeps source discovery, AEDAT parsing, interval segmentation and subject split in its
  own adapter. Prepare only `trials_to_train.txt`; keep official test absent from development.
  Derive a subject-disjoint validation view from explicit subject IDs after profiling the archive.
  Disable horizontal flip because it changes the meaning of left/right gesture classes. Freeze no
  DVS-Gesture window, bin width or training recipe before the exhaustive real-data profile.
- **Why:** this tests real multi-dataset reuse without a plugin framework, preserves the released
  temporal precision and prevents test leakage or label-changing augmentation. It also avoids
  adopting a third-party presegmented copy whose timestamps are reduced to milliseconds.
- **Rejected:** one loader or launcher per experiment; sharing dataset-specific preprocessing
  values; using official test for recipe selection; automatic left/right flip without label remap;
  hard-coding a literature sample count despite known preprocessing disagreement.
- **Reversal condition:** if the official archive structure or exhaustive profile contradicts the
  verified format, stop before creating a recipe and revise the adapter from the observed files.
- **Refs:** <https://openaccess.thecvf.com/content_cvpr_2017/html/Amir_A_Low_Power_CVPR_2017_paper.html>,
  <https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/datasets/dvs128_gesture.py>.
