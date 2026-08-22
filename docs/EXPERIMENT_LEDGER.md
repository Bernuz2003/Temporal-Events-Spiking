# Experiment ledger

This ledger contains only runs that influence a scientific or engineering decision. It is not a copy
of `history.csv` and does not invent missing run metadata.

## Active DVS-Lip runs

None. No DVS-Lip loader, representation or training recipe is implemented at the current P0 state.

## Engineering validation runs

| Run ID | Commit | Config | Seed | Question | Result | Interpretation | Decision consequence | Artifact |
|---|---|---|---:|---|---|---|---|---|
| SMOKE-P0-20260821-193155 | `0b7c552` clean | `configs/smoke.yaml` | 7 | Does the bounded synthetic path complete train, checkpoint, holdout, profile, audit and provenance capture without official test data? | PASS on CPU; 35-test suite also PASS | Integration evidence only; accuracy 0.25/macro-F1 0.10 and zero firing in the profiled batch provide no convergence or scientific evidence | closes local P0-06/07/08/09/12/14; does not close historical or SMILIES gates | `artifacts/smoke_synthetic__20260821_193155__seed7/smoke_summary.json`; environment SHA-256 `8e024dd862d253affa0ec88071f3426d72fefc38a8b0ca03a90ca8c0ce33f4cc` |

The run used `official_test_used: false`, Python 3.10.20, Torch 2.12.1+cu130 and CPU. Its artifact
directory is intentionally unversioned; the row records the immutable commit/config/seed and hash
needed to distinguish it from a scientific experiment.

## Imported historical evidence

| Run ID | Commit | Config | Seed | Question | Result | Interpretation | Decision consequence | Artifact |
|---|---|---|---:|---|---|---|---|---|
| LEGACY-DVSGC-EXPLORATORY | unknown | legacy order-2 baseline, exact resolved config unavailable here | unknown | Does the diagnostic model distinguish content and order? | Numerical observations are preserved in the archived protocol | Historical, single-checkpoint evidence; not a new verified run | informed D002/D003, but cannot select DVS-Lip architecture | [`archive/dvsgc/mechanistic_temporal_audit.md`](archive/dvsgc/mechanistic_temporal_audit.md) |

The row above deliberately records `unknown` rather than reconstructing provenance from prose. It
must not be promoted to a replicated result unless the original artifact and checkpoint are located
and hashed.

## Entry template

```text
Run ID:
Date:
Commit:
Dirty state:
Config:
Recipe ID:
Seed(s):
Dataset manifest hash:
Split manifest hash:
Question:
Predeclared interpretation:
Result:
Uncertainty:
Cost/state profile:
Interpretation:
Decision consequence:
Artifact:
Limitations:
```

Screening rows must be labeled `SCREENING`; only replicated confirmation may be labeled
`CONFIRMATORY`.
