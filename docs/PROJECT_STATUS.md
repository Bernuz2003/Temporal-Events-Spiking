# Project status

**Updated:** 2026-08-24
**Phase:** DVS-Lip baseline stabilization and DVS-Gesture real-data gate

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
- The first complete corrected-topology run (clean commit `811bdc0`, seed 42) reaches 30.62%
  validation accuracy and 29.19% Macro-F1. Its best epoch is 63/64 and validation loss reaches its
  minimum at epoch 64; this is valid screening evidence but not a converged frozen baseline.
- `configs/dvslip_e0.yaml` remains the sole active recipe. The next screening run changes only its
  horizon to 128 epochs and requires a fresh cosine schedule. New runs also write a resumable
  epoch-boundary `last.pt`; AMP overflow frequency no longer contaminates the finite gradient mean.
- D021 now aligns all residual paths with the official QKFormer formulation: Transformer residuals
  are direct, SPEDS branches spike before direct addition, and the unused non-spiking Conv-BN helper
  is removed. The corrected 1,000-step overfit reaches 100% final same-subset accuracy with loss
  0.894, finite gradients and a clean provenance record; this was the gate that authorized the
  completed 64-epoch screening run.
- DVS-Lip remains the architecture-development benchmark. DVS-Gesture, DailyDVS-200 and CIFAR10-DVS
  will compare only their frozen baseline with the final DVS-Lip-selected architecture.
- The real DVS-Gesture archive is now present under `data/DvsGesture/`. Its official MD5 is verified;
  the 122 recordings reproduce the released disjoint lists (98 train, 24 test). Preparation remains
  strictly limited to train. The extracted source and compressed archive are retained together only
  until preparation/profile validation succeeds.
- Ruff, bytecode compilation, shell syntax and diff checks pass locally. The local Python lacks
  PyTorch/NumPy/pytest, so checkpoint tests and real DVS-Gesture preparation remain to be verified in
  the SMILIES container.

## Open gate

1. Commit the current implementation and pass the generic DVS-Lip gate with the existing image.
2. Launch the fresh 128-epoch DVS-Lip run; use `last.pt` only if that run is interrupted.
3. Prepare and exhaustively profile the official DVS-Gesture train archive without opening its
   official test split.

## Next task

Review the real DVS-Gesture profile, then choose its development subjects, physical-time E0 bins and
baseline recipe before adding a runnable training configuration.
