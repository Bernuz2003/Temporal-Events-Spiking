# Project status

**Updated:** 2026-09-02
**Phase:** final DVS-Lip capacity point running; P2 controls implementation-ready

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
- The recovered clean run `dvslip_e0__20260825_211710__seed42` at `9393b3c` is the canonical
  500,708-parameter E0
  development baseline at its Macro-F1-selected epoch 112: 44.81% validation accuracy, 44.15%
  Macro-F1, paper-semantic development Acc1 37.07% and Acc2 52.54%. Validation loss reaches its
  minimum at epoch 126 and the final window is flat, so extending the exhausted cosine schedule is
  not justified. This is single-seed, sample-stratified and non-speaker-disjoint development
  evidence, not a literature-comparable official-test score; the official test is untouched.
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
- The aligned 1,113,508-parameter capacity run reaches 49.58% accuracy and 49.38% Macro-F1,
  improving the 500k baseline by 4.77/5.22 points while more than doubling potential operations.
  Its larger train/validation gap is a real generalization limitation, not evidence that longer
  training is needed. The predeclared 1,967,972-parameter point is running and remains the final
  capacity observation; it does not delay the already selected 500k development track.
- D026 selects the 500,708-parameter model as the P2/P3 development scale because compactness and
  experiment throughput are thesis objectives. The 1M/2M results remain capacity controls; only
  final or shortlisted improvements are transferred to the larger scale.
- Compactness is not represented by parameters alone: the unquantized 500k profile has 16.02 Mbit
  of parameters but 35.19 Mbit of persistent LIF membrane state (1,099,776 elements, 2.20x parameter
  bits). State and traffic must therefore accompany parameters in later Pareto comparisons.
- P2 uses the same model and training path rather than parallel experiment code. `NoCrossTime`
  resets LIF state at every bin; readout is selectable among temporal mean, last state and a
  six-parameter-per-channel diagonal gated recurrence. The profiler distinguishes LIF and gated
  state, counts gated inference operations and reports observed update gates. Readout time is now
  explicit: `fixed_window` processes the complete causal horizon, while `last_event` returns the
  latest occupied-bin snapshot so trailing silence cannot confound the first readout comparison.
- The P2-01 shortcut command compares aligned and order-invariant views of exactly the same
  per-bin OFF/ON counts. The experimental control is deliberately separate from the ordinary data
  gate. Its first clean artifact reaches 6.34% aligned versus 4.97% order-invariant validation
  accuracy: bin position adds a small shortcut floor, but neither control approaches the 44.81%
  neural baseline. Both temporal fits reached their 100-iteration limit, so schema v2 records the
  final gradient norm and permits 300 iterations before the result is closed. The same
  resolved-config path can vary only E0 bin width for the later coarse/fine 2x2.
  Training-only temporal masking and time-consistent spatial erasing are available for a later
  bounded generalization check; both are disabled by default and never applied to validation.
- Ruff, bytecode compilation, shell syntax and diff checks pass locally. The complete test suite
  requires the project container because the local interpreter lacks the ML dependencies.

## Open gate

1. Pass the complete test suite after the compact convergence-report change and regenerate the
   P2-01 temporal control as schema v2.
2. Interpret the running 2M result as capacity evidence without extending the scan or delaying P2.
3. Run 500k `NoCrossTime` at fixed-window mean, then compare mean/last/gated at `last_event`. Gate
   the diagonal-gated full diagnostic on its bounded same-subset overfit check; test fixed-window
   gated only if the diagnostic shortlists it.
4. Treat generalization as validation improvement, not gap minimization. After P2 selects the
   temporal/readout baseline, screen temporal masking and spatial erasing one variable at a time;
   retain them only when validation Macro-F1 improves without degrading physical-time pAUC.

## Next task

Regenerate the compact P2-01 artifact and launch the independent 500k fixed-window `NoCrossTime` and
`mean@last_event` controls while the final capacity result is pending.
