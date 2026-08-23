# Project status

**Updated:** 2026-08-23
**Phase:** DVS-Lip foundation

## Current state

- The completed DVS-GC temporal/mechanistic audit remains executable but frozen. It is retained for
  regression and provenance; new DVS-Lip code must not depend on its dataset factory or audit
  orchestrators.
- Active implementation lives under `src/etsr/dvslip/`. Shared utilities are reused only when they
  match the new raw-event contract without adaptation to legacy assumptions.
- The official-train preflight passes on 100 classes and 14,896 samples without accessing test.
- The complete official-train content hash is
  `19307051df73845de2a19df0f718c554702e94f81835c7859b92f42ec56ffda4`.
- The deterministic class-stratified development split contains 11,901 train and 2,995 validation
  samples. Speaker identity is unavailable, so development validation is explicitly not
  speaker-disjoint.
- `EventSample` and the train-only raw `DvsLipDataset` preserve original event arrays, physical
  timestamps, binary polarity and stable IDs. The official test remains logically embargoed but
  does not require physical relocation (D013).
- The raw-loader/profile milestone passes its verified tests. The real profile validates every
  sample and records median duration 1.079 s, p95 1.358 s and maximum 1.889 s.
- E0 is implemented as the explicit D014 physical-time count encoder and adapts to the shared
  training engine without exposing an official-test holdout. The exhaustive 14,896-sample check
  preserves every event and observes maximum voxel count 17, so the uint8 gate is closed.
- Candidate recipe `dvslip_e0_r0` is implemented in `configs/dvslip_e0_recipe_r0.yaml` under the
  two-run maximum budget of D015. It adds only warm-up/cosine control, correct gradient accumulation
  and train-only horizontal flip. No DVS-Lip training has been run and the recipe is not frozen.
- The expanded shortcut/provenance suite passes all 60 tests inside the SMILIES image. PyTorch
  2.2.2 emits one non-blocking `SequentialLR` deprecation warning from its own milestone hand-off;
  the tested learning-rate trajectory remains correct.
- D016 records the verified physical-duration shortcut risk. The fixed global-statistic control,
  per-sample post-run diagnostics, dataset-index provenance and CUDA peak-memory logging are
  implemented; their expanded test suite and real shortcut artifact are not yet verified.
- The implementation is committed and the regenerated schema-3 preflight is ready, has no protocol
  blockers and reproduces the complete content hash over all 14,896 train samples. No important run
  may start from a dirty worktree.
- The single-home SMILIES workflow is prepared around the repository root: active checks, generic
  config-driven training and frozen DVS-GC helpers are separated under `scripts/`. The SIF and CUDA
  path are verified on the server RTX A4000. The first complete gate stopped only because the
  container's older Ruff still enabled the since-removed UP038 rule; the source now supports both
  lint versions without changing runtime behavior.
- Training execution is already dataset-agnostic at the YAML/launcher level. Dataset construction
  is not yet generic: D017 defers extraction of a neutral raw-event interface until DailyDVS-200
  supplies the second concrete active contract, avoiding both runner branches and speculative APIs.

## Open gate

- Rerun the complete server gate from the clean compatibility-fix commit, then run the one-epoch
  cost pilot.

## Next implementation task

Measure the shortcut floor and one-epoch runtime/peak CUDA memory before selecting the r0
micro-batch. Only then run `dvslip_e0_r0` with seed 42 and inspect its artifacts before freeze.

## Deferred external operations

Historical DVS-GC sanity data and the intended GitLab remote remain external inputs.
