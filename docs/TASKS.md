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
| P0-06 | Run local static quality baseline | PARTIAL | P0-02 | ruff/compile/shell/diff pass; pytest unavailable locally |
| P0-07 | Restore a bounded generic smoke workflow | PARTIAL | D008 | static checks pass; successful owner-run smoke artifact pending |
| P0-08 | Execute full test suite in supported environment | BLOCKED | Python 3.10–3.12 + dependencies | `pytest -q` result recorded |
| P0-09 | Verify no cross-sample LIF state leakage | PARTIAL | P0-08 | focused tests added; suite pass pending |
| P0-10 | Run historical DVS-GC sanity benchmark | BLOCKED | dataset/checkpoint/runtime | command, commit, config and plausible result |
| P0-11 | Verify Singularity, CUDA and environment logging on SMILIES | BLOCKED | SMILIES access | build/run record and environment artifact |
| P0-12 | Define active/frozen code boundary for `mechanistic_*` | PARTIAL | P0-07–P0-09 | policy/annotations complete; regression result pending |
| P0-13 | Reconcile the intended SMILIES GitLab integration remote | BLOCKED | owner/remote details | verified GitLab remote and non-destructive branch policy |
| P0-14 | Add machine-readable environment capture to run artifacts | PARTIAL | P0-02 | code/static checks complete; runtime artifact and pytest pending |

`PARTIAL` in P0-06 is intentional even though the other four static commands pass: the current
interpreter is Python 3.14.6 and lacks Torch, NumPy, PyYAML and pytest, so compilation does not prove
runtime correctness.

## P1 — foundation

| ID | Task | Status | Depends on | Required evidence / artifact |
|---|---|---|---|---|
| P1-01 | Deep-read required primary papers and official repositories | PENDING | P0 gate | completed source records with claim boundaries |
| P1-02 | Complete DVS-Lip novelty matrix | PENDING | P1-01 | `novelty_matrix.md` with verified cells |
| P1-03 | Complete DVS-Lip Pareto table | PENDING | P1-01 | `dvslip_pareto.md` with comparability caveats |
| P1-04 | Verify official DVS-Lip layout, license and speaker protocol | PENDING | P0 gate | source-backed protocol note |
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
