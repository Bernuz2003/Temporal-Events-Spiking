# Project status

**Updated:** 2026-08-26
**Phase:** frozen E0 baselines and DVS-Lip capacity gate

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
- The clean 128-epoch run `dvslip_e0__20260824_114511__seed42` freezes the 500,708-parameter E0
  development baseline at its Macro-F1-selected epoch 112: 44.81% validation accuracy, 44.15%
  Macro-F1, paper-semantic development Acc1 37.07% and Acc2 52.54%. Validation loss reaches its
  minimum at epoch 126 and the final window is flat, so extending the exhausted cosine schedule is
  not justified. This remains single-seed development evidence; the official test is untouched.
- The global-statistic shortcut reaches only 2.64% validation accuracy. Best-checkpoint correctness
  nevertheless correlates weakly with duration (0.111) and active-bin count (0.109), so physical
  duration is not the main explanation for 44.81% but remains a controlled P2 caveat.
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
  ms bins (100 steps), covering the observed 18.457 s maximum without duration normalization. Its
  exhaustive server scan preserves all 425,993,547 events and observes maximum voxel count 167,
  safely below the uint8 cap 255.
- Clean run `dvsgesture_e0__20260825_213021__seed42` freezes the 489,227-parameter DVS-Gesture
  baseline at epoch 117: 84.47% speaker-disjoint development accuracy and 83.62% Macro-F1. Training
  reaches 100% while the validation plateau remains near 82–84%; the main confusions are opposing
  rotation directions and `air_guitar` versus `other_gestures`. This is a credible transfer
  baseline, not an official-test or multi-seed claim.
- Shared final evaluation reports two non-interchangeable temporal curves: causal physical-time
  latency in microseconds and an utterance-relative diagnostic whose use of the final duration is
  explicitly marked as oracle endpoint knowledge. Both report interval-normalized accuracy AUC.
  DVS-Lip additionally computes source-defined Acc1/Acc2 and direct substitutions within the 25
  declared confusable pairs. These metrics run only after restoring the selected best checkpoint;
  overfit checks do not pay their cost.
- Two capacity-scan configs preserve the complete frozen DVS-Lip protocol and change only model
  width: `embed_dim=192` (1,113,508 parameters) and `embed_dim=256` (1,967,972 parameters).
- Ruff, bytecode compilation, shell syntax and diff checks pass locally. The complete test suite
  requires the project container because the local interpreter lacks the ML dependencies.

## Open gate

1. Pass the complete test suite in the clean server/container gate.
2. Run the predeclared approximately 1M/2M capacity sanity check with the frozen E0 recipe before
   attributing the remaining accuracy gap to architecture or temporal representation.
3. Interpret capacity together with parameter count, runtime and both explicitly labelled temporal
   curves; freeze the smallest adequate width before starting P2 architecture controls.

## Next task

Run the two DVS-Lip capacity controls one at a time on the RTX A4000, then select the capacity used
for P2 without changing the frozen training recipe.
