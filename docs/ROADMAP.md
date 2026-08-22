# Roadmap

**Active phase:** P0
**Core thesis:** P0–P5
**Stretch:** P6 only after the core thesis is complete

This document is the phase-level view. Executable work and exact status live in
[`TASKS.md`](TASKS.md); scientific details remain in sections 18–33 of
[`PROJECT_CHARTER.md`](PROJECT_CHARTER.md).

## Phase map

| Phase | Question / outcome | Entry gate | Exit gate | Status |
|---|---|---|---|---|
| P0 | Trustworthy repository transition | DVS-GC audit snapshot identified | tests, docs, smoke, state reset and environment verified | **ACTIVE** |
| P1 | Reproducible raw DVS-Lip baseline, recipe and capacity sanity | P0 closed | loader/split/profile, Pareto, frozen recipe, 0.5/1/2M pilot | PENDING |
| P2 | Benchmark validity and architecture sanity | stable E0 recipe | shortcut controls, temporal dependency, readout test and central 2×2 | PENDING |
| P3 | Compact representation/core selection | interpretable P2 result | screened E0–E5 and replicated finalists with fair cost controls | PENDING |
| P4 | Hardware-aware consolidation | final architecture selected | quantization, state/memory profile and streaming equivalence | PENDING |
| P5 | DailyDVS-200 transfer | frozen DVS-Lip mechanism | rigid and single-`alpha` transfer reported | PENDING |
| P6 | Predictive/change-driven stretch | core thesis complete | only evidence-backed optional result | DEFERRED |

## P0 — repository transition and reproducibility

Goal: make repository state and instructions trustworthy before adding DVS-Lip.

Completed in the documentation-transition iteration:

- tagged `806c0aa` as `dvsgc-audit-complete-2026`;
- created `developer` from the tagged commit;
- inspected code, scripts, configs, tests and legacy documents;
- archived DVS-GC narrative documents and notebook without deleting provenance;
- introduced active charter navigation, rules, decisions, task tracking and technical-state records.

Implemented and verified locally in the supported project environment:

- bounded synthetic `etsr smoke` / `make smoke` path;
- explicit repeated-call and batch-isolation tests for `MultiStepLIF`;
- allow-listed `environment.json` capture for every training run;
- frozen DVS-GC dependency/retirement policy.

The owner-run suite passed all 35 tests. The bounded CPU smoke passed at clean commit `0b7c552` and
produced the required environment, checkpoint, profile and audit artifacts. Its chance-level result
and zero profiled firing are expected to carry no scientific claim.

Still required before P0 may close:

- run one historical sanity benchmark with available data/checkpoint and record it;
- verify the Singularity build, CUDA and environment capture on SMILIES;
- reconcile the charter's SMILIES GitLab workflow with the currently configured GitHub-only remote;
- verify the new `environment.json` artifact on GPU when SMILIES is available.

Go/no-go: **NO-GO for DVS-Lip implementation until the remaining P0 items are complete or explicitly
waived in `DECISIONS.md`. Primary-source review may proceed under D009 because it changes no runtime
contract and reduces uncertainty before implementation.**

## P1 — DVS-Lip foundation

Ordered outcomes:

1. verified official dataset/protocol and speaker-disjoint manifest;
2. raw-event sample contract and deterministic loader;
3. `dataset_profile.json` used to select physical temporal scales;
4. primary-source review, novelty matrix and Pareto table;
5. bounded E0 recipe stabilization and freeze;
6. one-seed capacity scan near 0.5M, 1M and 2M.

The official test remains embargoed. Parameter-cap revision requires owner approval and a decision
record. S001, S003 and S006 are now source-verified; all three public implementations use the
official test during model development, and none supplies the sample-to-speaker mapping needed for
the project's 24/6 train/validation manifest. Dataset implementation therefore remains blocked by
the acceptance gate in [`DVSLIP_PROTOCOL.md`](DVSLIP_PROTOCOL.md), while the remaining source
review is deferred until an active implementation task requires it under D010. The train-only
preflight, semantic class manifest and six regressions are implemented; the next step is running
that tooling against the real archive and supplied protocol metadata.

## P2 — validity and interaction

Run cheap time-resolved versus order-invariant baselines, a minimal NoCrossTime control, and
mean/last/compact-causal-readout comparison. Then confirm the declared 2×2:

```text
coarse vs fine representation × current LIF vs explicit compact temporal core
```

The interaction result determines whether P3 emphasizes representation, temporal state, their
co-design, or a baseline/capacity correction.

## P3 — compact selection

Screen E0–E5 under the frozen recipe and validated core. Report native parameters, operations,
state, temporal steps and compute. Replicate finalists across three seeds and add approximately
iso-parametric plus relevant iso-state/energy controls. Learned delays are conditional, not an
automatic candidate.

## P4 — consolidation

Quantize weights and temporal state, verify offline/chunked/step equivalence, extend profiling to
state traffic/buffers and complete hardware cards. Change-driven gating is optional only after the
base architecture is stable.

## P5 — transfer

Freeze topology, representation/core family, state components, precision and compression. Evaluate
DailyDVS-200 with `alpha=1` and one validation-selected global temporal calibration `alpha*`; do not
retune each time constant independently.

## P6 — stretch

Future prediction begins with trivial predictors and is abandoned if they are not beaten. A full
JEPA/predictive-coding direction is outside the core thesis and must not delay P0–P5.
