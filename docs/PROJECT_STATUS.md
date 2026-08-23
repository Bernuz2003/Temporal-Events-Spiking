# Project status

**Updated:** 2026-08-23
**Phase:** DVS-Lip baseline run and multi-dataset baseline foundation

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
- D021 now aligns all residual paths with the official QKFormer formulation: Transformer residuals
  are direct, SPEDS branches spike before direct addition, and the unused non-spiking Conv-BN helper
  is removed. The corrected 1,000-step overfit reaches 100% final same-subset accuracy with loss
  0.894, finite gradients and a clean provenance record, so the complete DVS-Lip E0 run is
  authorized without another recipe change.
- DVS-Lip remains the architecture-development benchmark. DVS-Gesture, DailyDVS-200 and CIFAR10-DVS
  will compare only their frozen baseline with the final DVS-Lip-selected architecture.
- DVS-Gesture is the second concrete dataset adapter. The working tree now has one shared raw-event
  boundary and encoded adapter, an official-AEDAT train-only preparation path, subject-disjoint
  development views and dataset-driven SMILIES prepare/gate commands. No DVS-Gesture representation
  or recipe values are frozen before the real archive profile.
- Ruff, bytecode compilation, shell syntax and diff checks pass locally. The local Python lacks
  PyTorch/NumPy/pytest, so the complete suite and real DVS-Gesture preparation remain unverified in
  the SMILIES container.

## Open gate

1. Run the authorized complete DVS-Lip E0 training from the clean D021 commit.
2. Verify the new test suite in the rebuilt SMILIES image.
3. Prepare and exhaustively profile the official DVS-Gesture train archive without opening its
   official test split.

## Next task

Review the real DVS-Gesture profile, then choose its development subjects, physical-time E0 bins and
baseline recipe before adding a runnable training configuration.
