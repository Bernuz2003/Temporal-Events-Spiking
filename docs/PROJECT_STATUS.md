# Project status

**Updated:** 2026-08-22
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
- The recipe implementation gate passes 56 unit tests and the bounded synthetic end-to-end smoke.
- D016 records the verified physical-duration shortcut risk. The fixed global-statistic control,
  per-sample post-run diagnostics, dataset-index provenance and CUDA peak-memory logging are
  implemented; their expanded test suite and real shortcut artifact are not yet verified.
- The current preflight artifact is schema 2 and must be regenerated as schema 3 after the current
  implementation is committed. No important run may start from a dirty worktree.

## Open gate

- Verify the expanded tests, commit the implementation, regenerate schema-3 preflight, then run the
  shortcut and one-epoch cost gates.

## Next implementation task

Measure the shortcut floor and one-epoch runtime/peak CUDA memory before selecting the r0
micro-batch. Only then run `dvslip_e0_r0` with seed 42 and inspect its artifacts before freeze.

## Deferred external operations

Historical DVS-GC sanity data, SMILIES/CUDA verification and the intended GitLab remote remain
external inputs. They are revisited only when an experiment or deployment actually requires them.
