# Training recipe

**Status:** DVS-Lip recipe not defined or frozen
**Owner decision required to change freeze after P1-10:** yes

## Purpose

Training is a controlled experimental variable. Architecture or representation conclusions are
invalid while optimizer, schedule, augmentation and regularization are moving simultaneously.

## Legacy implemented recipe — not a DVS-Lip recipe

The current YAML configs and engine implement:

```text
optimizer: AdamW
learning rate: 1e-3
weight decay: 5e-4
schedule: cosine over all epochs, no warmup
label smoothing: 0.1
gradient clipping: 1.0
AMP: enabled only on CUDA
checkpoint selection: configured metric (Macro-F1 in canonical DVS-GC configs)
augmentation: none in the engine
```

These are **FACTS about legacy code**, not a recommendation for DVS-Lip. The engine has no explicit
recipe ID, warmup, event-coordinate augmentation, temporal masking or recipe artifact.

## P1 bounded stabilization protocol

To be specified after S001/S003/S005/S006 and the raw dataset profile are verified. The plan must
declare before the first tuning run:

- baseline architecture/representation E0;
- train/validation manifest hashes;
- maximum number of configurations and seeds;
- optimizer and learning-rate candidates;
- warmup/scheduler candidates;
- weight-decay and label-smoothing candidates;
- augmentation candidates with probability/range and effects on time, count and geometry;
- stopping rule and selection metric;
- compute budget.

Do not copy a literature recipe blindly when event representation, batch semantics or model scale
differ.

## Recipe record template

```text
Recipe ID:
Status: CANDIDATE | FROZEN | SUPERSEDED
Date:
Commit:
Dataset/split hash:
Representation:
Architecture/capacity:
Optimizer:
Learning rate:
Warmup:
Scheduler:
Epochs:
Batch size / accumulation:
Weight decay:
Label smoothing:
Gradient clipping:
AMP/precision:
Spatial augmentations:
Temporal augmentations:
Event-rate/count augmentations:
Selection metric:
Search budget used:
Evidence/ledger runs:
Freeze decision:
Reversal condition:
```

## Freeze rules

After P1-10, controlled representation/core comparisons reuse the exact recipe and manifest. A
change requires a decision record and invalidates direct attribution unless all affected conditions
are rerun. Validation/test receive no stochastic augmentation. The official test cannot influence
the recipe.
