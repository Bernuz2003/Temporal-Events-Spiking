# Agent operating rules

**Scope:** all implementation, review, experiments and documentation on `developer`
**Authority:** derived from [`PROJECT_CHARTER.md`](PROJECT_CHARTER.md); it does not replace it.

## 1. Working posture

Act as a senior research engineer and co-owner of scientific quality. Reconstruct context before
editing, inspect the implementation instead of inferring it from prose, challenge weak assumptions
and preserve uncertainty. Activity is not evidence of progress if it cannot change a documented
decision.

The following labels are mandatory in research notes:

```text
FACT          directly verified
INFERENCE     conclusion derived from named facts
HYPOTHESIS    falsifiable proposition
DECISION      selected option with rationale and reversal condition
OPEN QUESTION missing information able to change the plan
```

## 2. Before changing code

1. Read the charter, `PROJECT_STATUS.md` and only the decisions relevant to the active task.
2. Inspect affected code, tests, configuration, scripts and Git history.
3. Read the primary paper and official implementation when reproducing an external mechanism.
4. State the scientific question and what each possible result would change.
5. Identify split, capacity, recipe, causality, state and hardware confounds.

Review sources just in time: read and update the entries required by the active external mechanism
immediately before implementing it; do not bulk-review unrelated future methods. Do not approve a
final architecture or numerical success threshold until the required novelty matrix and DVS-Lip
Pareto anchors are complete.

## 3. Experimental validity

- Change one principal variable per attribution experiment. The declared 2×2
  representation/core interaction is the deliberate exception.
- Freeze the E0 training recipe before representation or core comparisons.
- Select checkpoints and design choices on train/validation only.
- Do not access the official DVS-Lip test before the documented freeze decision.
- Report native cost in screening; use approximately iso-parametric and, when meaningful, iso-state
  or iso-energy comparisons for finalists.
- Treat one/two seeds as screening only. Decisive results require three seeds and uncertainty.
- Preserve valid negative results in `EXPERIMENT_LEDGER.md` and their consequences in
  `DECISIONS.md`.

Never conclude “time is irrelevant” from a failed richer representation until capacity, recipe,
preprocessing and temporal-core ability have been checked.

## 4. Architectural constraints

- Optimize for the finite thesis scope, not hypothetical product growth. Implement the simplest
  design that fully supports the required experiments and validity checks. Do not add extension
  points, registries, compatibility layers, manifests or generic abstractions for possible future
  use; complexity is justified only by a current requirement or a concrete next task.
- Final candidates are causal: no future event affects output/state at time `t`.
- Streaming state ownership must be explicit and tested (`reset_state`, `step`, sequence and chunk
  equivalence, detach semantics).
- Mini-QKFormer is a baseline; QKTA is only a candidate efficient spatial mixer.
- LIF is already a temporal recurrence. Any extra state must demonstrate a distinct capability or
  Pareto advantage.
- Bidirectional recurrence is literature evidence or an approved upper bound, never the final
  streaming mechanism.
- Do not create encoder/temporal abstractions until a real implementation requires them.

## 5. Data and failure behavior

Fail loudly on:

- timestamp clipping or duration normalization;
- corrupt-sample dropping;
- split substitution or official-test use;
- class-order changes;
- dependency substitutions;
- encoder fallbacks;
- manifest/checkpoint incompatibility.

Raw DVS-Lip events remain `(x, y, t, polarity)` until an explicit representation module. Physical
time, polarity convention and stable sample ID must be observable metadata. Speaker identity must
be represented only when authoritative metadata exists; otherwise its absence and the resulting
non-speaker-disjoint development split must be explicit.

## 6. Hardware honesty

Binary spikes are not an energy result. Every candidate reports parameters, MAC, AC/SOP, persistent
state bits, reads/writes, buffer depth, precision, nonlinear/LUT cost, sparsity assumptions and
feedback path. Horowitz arithmetic energy is a proxy and must never be called measured FPGA energy.

A hardware card is required before implementing spatially distributed recurrent state. Persistent
state cannot be FP32-only in the final candidate.

## 7. Reproducibility

Every important run records:

```text
commit and dirty state
resolved config and recipe ID
seed
dataset and split hashes
environment
parameter count and state profile
artifact path
```

Before a milestone:

```bash
pytest -q
ruff check src tests
python -m compileall -q src tests
make check-scripts
git diff --check
```

Missing dependencies or unavailable hardware are recorded as blockers, never reported as a pass.

## 8. Documentation discipline

- `PROJECT_STATUS.md` is the only routinely updated document; keep only current outcome, blockers
  and next task.
- `DECISIONS.md` receives only choices that materially change future scientific or structural work.
- All other documents are read-only unless their revision is the explicitly approved task.
- Do not mirror diffs, test logs, command history or implementation details already recoverable from
  code, Git or artifacts.
- Add a machine-readable artifact only when runtime code consumes it, it freezes an assignment that
  must reproduce exactly, or a gate cannot be audited without it. Keep rationale and owner-provided
  provenance in the two writable documents; prefer one self-contained artifact per responsibility.

## 9. Stop and escalate

Ask the scientific owner before changing dataset, split, parameter target, main objective,
architecture family, test embargo or success criteria. Also stop when primary sources disagree, the
baseline remains far below anchors after recipe stabilization, a representation comparison is
capacity-confounded, streaming equivalence fails, or a design is not plausibly FPGA-mappable.

Autonomous actions within scope include tests, assertions, logging, profiling, documentation repair,
API cleanup and eliminating a clearly dominated screening candidate with recorded evidence.

## 10. Definition of task completion

A task is `DONE` only when:

1. every declared artifact exists;
2. acceptance checks have been executed in a suitable environment;
3. failures and unavailable checks are explicit;
4. `PROJECT_STATUS.md` is updated only if outcome, blockers or next task changed;
5. materially deferred work has a reason;
6. the next decision is unambiguous.

“Implemented” without verification is `IN PROGRESS`; “cannot run here” is `BLOCKED` or `PARTIAL`, not
`DONE`.
