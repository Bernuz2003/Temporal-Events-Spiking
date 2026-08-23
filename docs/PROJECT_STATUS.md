# Project status

**Updated:** 2026-08-23
**Phase:** DVS-Lip baseline stabilization

## Current state

- The approved deep cleanup leaves one active DVS-Lip pipeline. Historical DVS-GC code is removed
  from the working runtime and remains recoverable from Git history and the archived documentation.
- The official-train preflight passes on 100 classes and 14,896 samples without accessing test. Its
  complete content hash is
  `19307051df73845de2a19df0f718c554702e94f81835c7859b92f42ec56ffda4`.
- The deterministic development split contains 11,901 train and 2,995 validation samples. Speaker
  identity is unavailable, so validation is explicitly not speaker-disjoint.
- E0 uses 40 two-polarity count frames over a fixed 2 s physical-time window. The exhaustive scan
  preserves every event and observes maximum voxel count 17; the official test remains embargoed.
- `configs/dvslip_e0.yaml` is the sole active recipe. The failed steep surrogate was removed and the
  active hard-spike model uses the logistic backward surrogate with alpha 4.
- The first 100-step overfit remained collapsed at 6.25% final accuracy. The corrected-duration
  1,000-step run learned partially, reaching 59.38% best and 56.25% final accuracy on the same 64
  samples, but failed the 95%/1.5 gate. Its gradients were finite and clipping was active on every
  step from epoch 150 onward.
- D021 now aligns all residual paths with the official QKFormer formulation: Transformer residuals
  are direct, SPEDS branches spike before direct addition, and the unused non-spiking Conv-BN helper
  is removed. No recipe hyperparameter changed.
- Static local checks pass. The local Python installation has no PyTorch/pytest, so the complete
  cleaned test suite and the corrected forward/backward path must be verified in the SMILIES
  container before training.

## Open gate

1. Synchronize the D021 commit to the server and run `make smilies-dvslip-gate` from a clean
   worktree.
2. Run the fixed 16-class by 4-sample overfit for 500 epochs (1,000 optimizer steps).
3. Authorize complete E0 training only at final same-subset accuracy at least 95%, loss below 1.5
   and finite gradient metrics.

## Next task

Review the corrected residual overfit artifact; do not launch the complete E0 run beforehand.
