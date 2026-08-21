# Experiment ledger

This ledger contains only runs that influence a scientific or engineering decision. It is not a copy
of `history.csv` and does not invent missing run metadata.

## Active DVS-Lip runs

None. No DVS-Lip loader, representation or training recipe is implemented at the current P0 state.

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
