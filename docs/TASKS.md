# Task register

**Status vocabulary:** `DONE`, `IN PROGRESS`, `PARTIAL`, `PENDING`, `BLOCKED`, `DEFERRED`
**Rule:** status is evidence-based; an unavailable required check prevents `DONE`.

## P0 — active

| ID | Task | Status | Depends on | Required evidence / artifact |
|---|---|---|---|---|
| P0-01 | Tag the completed DVS-GC snapshot and create `developer` | DONE | — | tag and branch both resolve to `806c0aa` |
| P0-02 | Inspect charter, docs, scripts, source, configs and tests | DONE | — | findings in `REPOSITORY_STATE.md` |
| P0-03 | Replace active DVS-GC narrative with DVSLIP documentation system | DONE | P0-02 | active docs index, links and final diff checks |
| P0-04 | Archive legacy DVS-GC documents | DONE | P0-02 | `docs/archive/dvsgc/README.md` and preserved files |
| P0-05 | Resolve untracked notebook lifecycle | DONE | P0-02 | frozen notebook under `notebooks/archive/dvsgc/` plus index/checksum |
| P0-06 | Run local static quality baseline | DONE | P0-02 | ruff/compile/shell/diff pass; supported-runtime suite passes |
| P0-07 | Restore a bounded generic smoke workflow | DONE | D008 | `smoke_synthetic__20260821_193155__seed7` passed |
| P0-08 | Execute full test suite in supported environment | DONE | Python 3.10–3.12 + dependencies | 35 tests pass at `0b7c552` |
| P0-09 | Verify no cross-sample LIF state leakage | DONE | P0-08 | focused regressions included in passing suite |
| P0-10 | Run historical DVS-GC sanity benchmark | BLOCKED | dataset/checkpoint/runtime | command, commit, config and plausible result |
| P0-11 | Verify Singularity, CUDA and environment logging on SMILIES | BLOCKED | SMILIES access | build/run record and environment artifact |
| P0-12 | Define active/frozen code boundary for `mechanistic_*` | DONE | P0-07–P0-09 | policy/annotations plus full-suite/smoke regression |
| P0-13 | Reconcile the intended SMILIES GitLab integration remote | BLOCKED | owner/remote details | verified GitLab remote and non-destructive branch policy |
| P0-14 | Add machine-readable environment capture to run artifacts | DONE | P0-02 | runtime artifact and SHA-256 verified |

## P1 — foundation

| ID | Task | Status | Depends on | Required evidence / artifact |
|---|---|---|---|---|
| P1-01 | Deep-read required primary papers and official repositories | PARTIAL | active implementation need + D010 | completed source records with claim boundaries |
| P1-02 | Complete DVS-Lip novelty matrix | PARTIAL | P1-01 | `novelty_matrix.md` with verified cells |
| P1-03 | Complete DVS-Lip Pareto table | PARTIAL | P1-01 | `dvslip_pareto.md` with comparability caveats |
| P1-04 | Verify official DVS-Lip layout, license and speaker protocol | BLOCKED | speaker metadata + dataset terms + P0 gate | `DVSLIP_PROTOCOL.md` acceptance gate |
| P1-05 | Define raw event and encoded representation contracts | PENDING | P1-04 | reviewed API contract and tests |
| P1-06 | Implement deterministic raw DVS-Lip loader | PENDING | P1-05 | loader tests and manifests |
| P1-07 | Create speaker-disjoint train/validation/test manifest | PENDING | P1-04 | exact IDs, hashes, embargo checks |
| P1-08 | Generate dataset profile | PENDING | P1-06/P1-07 | `dataset_profile.json` and interpretation |
| P1-09 | Implement E0 via an explicit encoder layer | PENDING | P1-05/P1-06 | encoder tests and profiling contract |
| P1-10 | Stabilize and freeze bounded E0 training recipe | PENDING | P1-09 | recipe ID, budget, ledger and decision |
| P1-11 | Run 0.5M/1M/2M capacity sanity scan | PENDING | P1-10 | aligned pilot runs and capacity decision |

## P2–P6 — queued

| ID | Task | Status | Depends on |
|---|---|---|---|
| P2-01 | Time-resolved and order-invariant shortcut controls | PENDING | P1 exit |
| P2-02 | Minimal NoCrossTime/NoTD dependency control | PENDING | P1 exit |
| P2-03 | Mean vs last vs compact causal readout sanity | PENDING | P1 exit |
| P2-04 | Confirm central coarse/fine × LIF/explicit 2×2 | PENDING | P2-01–P2-03 |
| P3-01 | Broad E0–E5 representation screen | PENDING | P2-04 decision |
| P3-02 | Replicate and cost-match finalists | PENDING | P3-01 |
| P3-03 | Select final architecture by evidence | PENDING | P3-02 |
| P4-01 | Quantization and fixed-point temporal state | PENDING | P3-03 |
| P4-02 | Streaming equivalence and complete state/memory profile | PENDING | P3-03 |
| P4-03 | Optional change-driven gating | DEFERRED | P4-01/P4-02 |
| P5-01 | DailyDVS-200 rigid and single-`alpha` transfer | PENDING | P4 exit |
| P6-01 | Future-prediction stretch with trivial controls | DEFERRED | core thesis complete |

## Update protocol

For every status change add the date and evidence link in the task notes below. Do not erase a
postponed task; change it to `DEFERRED` and record the reason.

### Task notes

- **2026-08-21 — P0-01:** `developer`, `dvsgc-audit-complete-2026` and the source branch all resolve
  to `806c0aa` at transition time.
- **2026-08-21 — P0-06/P0-08:** ruff, compileall and shell syntax succeed locally; pytest cannot be
  collected because the command and runtime dependencies are absent.
- **2026-08-21 — P0-10/P0-11:** deferred to an environment with project data and SMILIES tooling;
  no result is inferred.
- **2026-08-21 — P0-13:** the only configured remote is GitHub `origin`; a live read-only query
  found `master` and `refactor/mechanistic-temporal-audit` but no `developer`. No GitLab remote or push
  was created because the intended integration endpoint is not inferable from the repository.
- **2026-08-21 — P0-03:** 18 required artifacts exist; 34 Markdown files were checked with zero
  missing local links; all 12 legacy documents are byte-identical to their `806c0aa` originals;
  `git diff --check` includes new files and passes; no implementation/config/script change exists.
- **2026-08-21 — P0-07/P0-09/P0-12/P0-14:** implemented bounded `etsr smoke`, two LIF isolation
  regressions, allow-listed environment capture and the frozen-code policy. The tree now contains 35
  tests. Ruff, compileall, shell syntax, Make dry-runs and diff checks pass; runtime status remains
  partial until the owner executes pytest and smoke in the supported environment.
- **2026-08-21 — P0-06/P0-07/P0-08/P0-09/P0-12/P0-14:** owner-run `make test` passed all 35 tests
  in 3.08 s at clean commit `0b7c552`. Bounded run
  `smoke_synthetic__20260821_193155__seed7` passed training, holdout, profiling and behavioral audit
  on CPU. Its `environment.json` has verified SHA-256
  `8e024dd862d253affa0ec88071f3426d72fefc38a8b0ca03a90ca8c0ce33f4cc`; resolved provenance and
  summary carry the same hash, `official_test_used` is false, and all required artifacts exist.
  Firing activity was zero in the single profiled batch, so this closes integration gates only and
  is not evidence of learning or useful spiking dynamics.
- **2026-08-21 — P1-01:** started under D009 while external P0 blockers remain; review work may
  update source/protocol records but cannot authorize DVS-Lip implementation or experiments.
- **2026-08-21 — P1-01/P1-04:** S001 and S003 paper, supplement where available, and exact official
  code commits were reviewed. Public sources confirm 30/10 speaker-disjoint counts but provide no
  sample-to-speaker mapping; both official training loops select checkpoints on `test/`. Dataset
  license terms were not found, and official MSTP Acc1/Acc2 code appears reversed relative to paper
  semantics. Findings and the unblock gate are in [`DVSLIP_PROTOCOL.md`](DVSLIP_PROTOCOL.md).
- **2026-08-21 — P1-01/P1-02/P1-03:** S006 paper, supplement, Git history and official ALR code at
  `af3c320` were reviewed. NSA establishes strong temporal-mechanism sensitivity at approximately
  9.5M parameters, not at the preferred 500k scale. Six distinct rows now separate its controls;
  its ALR efficiency is unreported, its Transformer is non-causal, and its code selects on the
  official test. The supplement/code LTC-SFNN width mismatch remains an explicit reproduction
  caveat.
- **2026-08-21 — P1-01:** broad source review paused by D010 after S001/S003/S006. Remaining sources
  stay `LISTED` until the corresponding implementation/comparison task becomes active.
- **2026-08-21 — P1-04:** implemented train-only `preflight-dvslip`, strict structured-sample
  inspection, exact speaker/split/terms/hash validation and the S001-derived semantic class manifest.
  Six new tests bring the full suite to 41 passing in 4.36 s; Ruff, compileall and shell syntax pass.
  P1-04 remains `BLOCKED` because no local archive, authoritative sample-to-speaker map, dataset
  terms or physical official-test quarantine have been supplied and verified.
