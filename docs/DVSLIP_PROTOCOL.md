# DVS-Lip protocol record

**Updated:** 2026-08-21

**Status:** train-only preflight implemented and statically verified; loader implementation blocked
on the actual archive, speaker metadata, dataset-use terms and physical test quarantine

**Scope:** dataset layout, raw sample semantics, split policy, metric semantics and embargo

This record separates what the public sources establish from what still requires inspection of the
actual dataset. It is not a loader specification yet.

## Sources and exact versions

- Tan et al., CVPR 2022, main paper and official supplement (S001).
- Official MSTP repository `tgc1997/event-based-lip-reading`, commit
  `a0246506f4d4f3d15f187b18ae9c1010aa146cdb` (2022-06-19).
- Dampfhoffer and Mesquida, CVPRW/EVW 2024 (S003).
- Official SpikGRU repository `manondampfhoffer/SpikGRU-DVSLip`, commit
  `a61c6afcdf5f0398646a885ce416da690cf8351f` (2024-11-27).
- Li et al., IJCAI 2025 *Neuromorphic Sequential Arena* and official supplement (S006).
- Official NeuroSeqBench repository `liyc5929/neuroseqbench`, commit
  `af3c3204df0c3bedc9abfde3a01cc62f42effde5` (2025-08-14).

URLs and detailed claim boundaries are maintained in [`SOURCES.md`](SOURCES.md).

## Dataset facts established by the papers

- DVS-Lip contains 100 English word classes, 40 speakers and 19,871 valid word samples.
- The collection used DAVIS346 event/intensity output. The released word samples are described as
  mouth-centered 128×128 crops derived after audio-assisted word segmentation.
- Forty volunteers, 20 per gender, each read five randomized sequences containing the 100 words.
  Damaged recordings account for the difference from the nominal 20,000 samples.
- The official partition contains 14,896 samples from 30 speakers for training and 4,975 samples
  from 10 disjoint speakers for evaluation.
- Fifty classes form 25 visually confusable pairs (paper/supplement Part 1); the remaining 50 are
  common words (Part 2).
- A raw event is described semantically by `(x, y, t, p)`, with microsecond-resolution time and
  binary polarity. The released NumPy representation must still be inspected directly before its
  dtype, units and ordering become a project contract.

## Layout consumed by both official repositories

Both loaders expect this effective tree:

```text
DVS-Lip/
  train/<word>/<integer>.npy
  test/<word>/<integer>.npy
```

The MSTP loader reads structured fields `t`, `x`, `y`, `p`; the SpikGRU and NSA loaders convert
`np.load(...).tolist()` to an `N×4` array and treat its columns as `t`, `x`, `y`, `p`. This is code
evidence about the expected input, not a substitute for inspecting a downloaded sample.

The MSTP variable named `person` is only the integer file stem used to index a per-word
`frame_nums.json` array. Neither official repository derives or returns a speaker ID. The public
`frame_nums.json` contains 100 words and totals exactly 14,896 train plus 4,975 test entries, but it
contains frame counts—not speaker identity.

## Published representations

### MSTP

- Events are cropped from the 128×128 sample to coordinates `[16,112)²`, yielding 96×96.
- Polarity 0 is converted to −1 and accumulated into a single signed voxel grid.
- Timestamps are normalized per sample and linearly interpolated to adjacent temporal voxels.
- The paper result uses low/high temporal dimensions `(30,210)` and final 88×88 crops.
- The latest README test command realizes this as `num_bins=1+7`; its training command uses `1+4`,
  so it does not directly reproduce the paper's 210-bin configuration.

### SpikGRU2+

- Samples are cut at 1.2 s and discretized into 90 nearest-neighbor bins (about 13 ms per bin).
- Positive and negative polarities occupy separate channels; there is no temporal interpolation.
- The paper/code use 88×88 outputs and stronger spatial plus temporal masking augmentation.
- This is a synchronous training representation. A possible event-driven hardware mapping discussed
by the paper is not an implemented streaming loader or equivalence result.

### Neuromorphic Sequential Arena ALR

- Events are grouped into 200 nearest bins with separate positive/negative channels.
- Training uses central/random 88×88 crops, horizontal flip, cutout and random zoom; evaluation
  uses a central 88×88 crop.
- The loader eagerly reads the same `train|test/<word>/<integer>.npy` tree and exposes no speaker
  identity.
- The final published task readout uses the last temporal step.

These three encoders are literature baselines. None becomes the repository's canonical raw-event
contract merely because it appears in public code.

## Split and embargo finding

The papers establish speaker-disjoint official train/test counts, but the public repositories expose
no sample-to-speaker manifest and no speaker IDs. All three reviewed training scripts evaluate the
official `test/` tree every epoch and select the best checkpoint/result from that score. That
protocol is incompatible with this project's test embargo.

Therefore:

- `test/` must be quarantined before any exploratory data access or model selection;
- a 24-speaker train / 6-speaker validation split cannot be inferred from integer filenames,
  ordering, groups of five or missing-sample patterns;
- sample-level random splitting is prohibited because it would violate the speaker-disjoint intent;
- P1-04/P1-07 cannot close until an authoritative sample-to-speaker mapping is available and hashed.

## Acc1/Acc2 inconsistency

The main paper and supplement define:

- `Acc1`: accuracy on Part 1, the 50 visually confusable words;
- `Acc2`: accuracy on Part 2, the 50 common words.

They report MSTP `Acc1=62.17%`, `Acc2=82.07%`, overall `72.10%`, consistent with the first group
being harder. In official commit `a024650`, however, `compute_each_part_acc` assigns words *not* in
the `ambigious_labels` list to `acc_part1`, and the list exactly matches supplement Part 1. The code
labels therefore appear reversed relative to the paper.

Project policy:

- store semantic names `visually_confusable_accuracy` and `common_words_accuracy` internally;
- expose `Acc1`/`Acc2` only with the paper definitions and a unit-tested 50/50 class manifest;
- do not copy the public function without correcting and documenting this discrepancy.

The versioned [`dvslip_class_groups.json`](../configs/dvslip_class_groups.json) now records all 100
classes, the 25 confusable pairs and the semantic mapping `Acc1=visually confusable`, `Acc2=common`.
The preflight validates disjoint 50/50 coverage against the class directories; tests exercise the
mapping independently of a real dataset.

## Licensing boundary

The MSTP code repository contains Apache-2.0; the SpikGRU code repository contains CeCILL-B plus an
MIT notice for its imported Cutout function; NeuroSeqBench contains GPL-3.0. Those files license
source code. No dataset-specific license or terms were found in the paper, supplement, public README
or download page inspected in this review. Code licenses must not be represented as a license grant
for the recordings.

Dataset access/use is therefore an open evidence item, not an assumed permission. The dataset
archive may contain terms absent from the public pages; inspect and preserve them when data becomes
available.

## Implemented train-only preflight

`etsr preflight-dvslip` and `make preflight-dvslip` are deliberately separate from the legacy
frame factory. They accept only a path whose final component is `train`, reject symlinks and layout
drift, inspect deterministic representative structured arrays, and never discover or traverse a
`test/` sibling. The report always carries `official_test_used: false`.

The quick first pass is:

```bash
make preflight-dvslip DVSLIP_TRAIN_ROOT=/absolute/path/to/DVS-Lip/train
```

It should report external blockers rather than invent missing provenance. A complete evidence pass
also accepts:

```bash
make preflight-dvslip \
  DVSLIP_TRAIN_ROOT=/absolute/path/to/DVS-Lip/train \
  DVSLIP_SPEAKER_MANIFEST=/absolute/path/to/speakers.csv \
  DVSLIP_SPLIT_MANIFEST=/absolute/path/to/split.json \
  DVSLIP_TERMS=/absolute/path/to/DATASET_TERMS.txt \
  DVSLIP_HASH_SAMPLES=1
```

Full content hashing may be slow and is not run automatically. `speakers.csv` must exactly cover
official-train files with these columns:

```text
relative_path,speaker_id
word/0.npy,<authoritative-speaker-id>
```

`split.json` is owner/version controlled and must not be inferred from filenames:

```json
{
  "schema_version": 1,
  "official_source_split": "train",
  "official_test_used": false,
  "split_seed": 0,
  "strategy": "documented selection rule",
  "assignments": {"<speaker-id>": "train"}
}
```

The validator requires exactly 24 train and 6 validation speakers, complete speaker coverage and
zero overlap. A passing preflight is still not the full P1-04 gate: the report keeps
`protocol_gate_status: blocked` until physical official-test quarantine is independently verified.

## Acceptance gate before loader implementation

P1-04 may close only after all of the following are recorded:

1. local dataset source and archive/file hashes;
2. dataset-specific license/terms or an explicit owner-approved resolution of their absence;
3. representative sample dtype, shape, field order, time unit/range, polarity values and coordinate
   range inspected directly;
4. authoritative sample-to-speaker mapping for all official-train samples;
5. deterministic 24/6 speaker manifest with zero overlap and exact sample counts;
6. physical quarantine plus code-level embargo of the 10-speaker official test partition;
7. semantic Part 1/Part 2 class manifest and tested Acc1/Acc2 mapping.

Item 7 now has a versioned implementation and tests; it still needs comparison against the actual
archive directories. Until the whole list is verified, no DVS-Lip loader, split generator, training
run or success threshold is authorized.
