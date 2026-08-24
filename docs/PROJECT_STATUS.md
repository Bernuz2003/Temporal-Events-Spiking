# Project status

**Updated:** 2026-08-24
**Phase:** DVS-Lip baseline stabilization and DVS-Gesture E0 gate

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
- DVS-Gesture preparation and exhaustive raw-event validation pass on all 1,176 official-train
  gestures from 23 subjects; the official test remains unused. The deterministic hash-ranked
  validation subjects `[8, 11, 17, 19, 22]` give 912 train and 264 validation samples with
  near-exact class proportions. The sole candidate E0 config uses a 20 s physical window with 200
  ms bins (100 steps), covering the observed 18.457 s maximum without duration normalization.
- Ruff, bytecode compilation, shell syntax and diff checks pass locally. The complete test suite and
  exhaustive DVS-Gesture E0 count-cap scan remain server-side gates.

## Open gate

1. Let the active 128-epoch DVS-Lip run finish; use `last.pt` only if it is interrupted.
2. Exhaustively scan DVS-Gesture E0 for exact event preservation and uint8 count-cap safety.
3. Pass a bounded DVS-Gesture overfit before authorizing its complete baseline run.

## Next task

Run the DVS-Gesture E0 representation scan and bounded overfit without competing for the GPU used by
the active DVS-Lip run.
