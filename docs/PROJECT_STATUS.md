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
- The failed steep surrogate is removed. Its epoch-28 evidence remains in the ignored run artifact
  and Git history, not as a second supported recipe or diagnostic subsystem. The exact-forward
  checkpoint comparison showed that sigmoid alpha 4 recovers mean gradients by 66.1 million times
  in the first embedding and 6.56 million times in stage 1, while leaving logits unchanged.
- `configs/dvslip_e0.yaml` is the sole active E0 recipe. The model supports one logistic surrogate
  with configurable alpha, currently 4; there is no selector or compatibility default for the
  falsified implementation.
- A generic train-only overfit mode now reuses the normal runner on a deterministic balanced subset,
  disables DVS-Lip augmentation, AMP and profiling, and reports mean pre-clipping gradient norm and
  clipping frequency. No DVS-Lip-specific overfit loop or permanent runtime config was added.
- The previous SMILIES gate passed its then-current suite. The simplified 64-test suite must pass
  the next server gate; PyTorch 2.2.2 may still emit its non-blocking `SequentialLR` deprecation
  warning.
- D016 records the verified physical-duration shortcut risk. The fixed global-statistic control,
  per-sample post-run diagnostics, dataset-index provenance and CUDA peak-memory logging are
  implemented and verified on the real development split. The E0-observable global-statistic floor
  is 2.64% validation accuracy and 1.43% Macro-F1; it is a required comparison, not model evidence.
- The implementation is committed and the regenerated schema-3 preflight is ready, has no protocol
  blockers and reproduces the complete content hash over all 14,896 train samples. No important run
  may start from a dirty worktree.
- The single-home SMILIES workflow is prepared around the repository root: active checks, generic
  config-driven training and frozen DVS-GC helpers are separated under `scripts/`. The SIF and CUDA
  path are verified on the server RTX A4000. Training now supplies the cuBLAS workspace setting
  required by the declared deterministic CUDA mode; no image rebuild is required.
- Training execution is already dataset-agnostic at the YAML/launcher level. Dataset construction
  is not yet generic: D017 defers extraction of a neutral raw-event interface until DailyDVS-200
  supplies the second concrete active contract, avoiding both runner branches and speculative APIs.

## Open gate

- Run the fixed 16-class × 4-sample, 50-epoch overfit check on official train only. It passes at
  final-epoch same-subset accuracy at least 95% and loss below 1.5; clipping behavior must remain
  finite.

## Next implementation task

Inspect the overfit history and summary before authorizing the complete E0 training.

## Deferred external operations

Historical DVS-GC sanity data and the intended GitLab remote remain external inputs.
