# Source register

**Purpose:** track what has actually been read and which claims are allowed.
**Current state:** P1-01 partial under D010. S001, S003 and S006 have paper/code review records;
remaining sources stay listed until an active implementation/comparison task requires them.

## Status vocabulary

- `LISTED`: URL/title supplied by the charter; contents not yet independently reviewed.
- `PAPER READ`: primary paper and relevant supplement inspected.
- `CODE READ`: official implementation/version inspected.
- `VERIFIED`: both required evidence paths completed and notes recorded.
- `BLOCKED`: source unavailable or identity/provenance ambiguous.

`LISTED` is not evidence for a novelty, performance, parameter or hardware claim.

## Required DVS-Lip and temporal-model sources

| ID | Source | Primary URL | Official code | Status | What the project expects to use | Claims prohibited before verification |
|---|---|---|---|---|---|---|
| S001 | MSTP / original DVS-Lip, Tan et al., CVPR 2022 | <https://openaccess.thecvf.com/content/CVPR2022/papers/Tan_Multi-Grained_Spatio-Temporal_Features_Perceived_Network_for_Event-Based_Lip-Reading_CVPR_2022_paper.pdf> | <https://github.com/tgc1997/event-based-lip-reading> | VERIFIED | dataset/protocol, Acc1/Acc2, multi-rate representation | speaker IDs, dataset license or reproduced accuracy |
| S002 | TVTA | <https://arxiv.org/abs/2607.08236> | not supplied | LISTED | temporal modeling before spatial aggregation; novelty boundary | architecture details or comparable cost |
| S003 | SpikGRU2+ | <https://openaccess.thecvf.com/content/CVPR2024W/EVW/papers/Dampfhoffer_Neuromorphic_Lip-Reading_with_Signed_Spiking_Gated_Recurrent_Units_CVPRW_2024_paper.pdf> | <https://github.com/manondampfhoffer/SpikGRU-DVSLip> | VERIFIED | accuracy anchor, recurrence, augmentation, causality/scale caveats | gating-only attribution, compact feasibility or measured hardware energy |
| S004 | Multiplication-Free Channel-wise PSN | <https://papers.nips.cc/paper_files/paper/2025/file/5f40a99efedc1148a061d80201949c84-Paper-Conference.pdf> | not supplied | LISTED | shift/add temporal dynamics and DVS-Lip anchor | sub-500k or FPGA result |
| S005 | Event2Vec | <https://arxiv.org/abs/2504.15371> | <https://github.com/Intelligent-Computing-Lab-Panda/event2vec> | LISTED | event-native representation, long-memory behavior, recipe | exact parameter count inferred from storage |
| S006 | Neuromorphic Sequential Arena | <https://www.ijcai.org/proceedings/2025/0544.pdf> | <https://github.com/liyc5929/neuroseqbench> | VERIFIED | core/neuron sensitivity and trade-offs | sub-500k Pareto proof or Transformer rejection |
| S007 | Spiking Patches | <https://arxiv.org/abs/2510.26614> | <https://github.com/DTU-PAS/spiking-patches> | LISTED | representation prior art | direct overlap/novelty conclusion |
| S008 | FATE | <https://arxiv.org/abs/2606.17334> | not supplied | LISTED | representation prior art | implementation or performance details |

## Architecture and delay sources

| ID | Source | Primary URL | Official code | Status | Intended use |
|---|---|---|---|---|---|
| S009 | QKFormer | <https://arxiv.org/abs/2403.16552> | <https://github.com/zhouchenlin2096/QKFormer> | LISTED | QKTA/SSA reference and reshape/LIF issue check |
| S010 | RVT | <https://arxiv.org/abs/2212.05598> | not supplied in charter | LISTED | spatial attention plus temporal recurrence prior art |
| S011 | TEFormer | <https://arxiv.org/abs/2601.18274> | not supplied | LISTED | temporal-enhancement novelty boundary and bidirectionality caveat |
| S012 | BAM-SLDK | <https://doi.org/10.1088/2634-4386/addb6c> | not supplied | LISTED | delayed kernels, pruning and hardware awareness |
| S013 | PCL+ | <https://arxiv.org/abs/2605.12732> | not supplied | LISTED | delay plus predictive-coding novelty boundary |

## Hardware/quantization sources requiring exact identification

The charter names the following but does not provide a unique primary URL/version. Their identity
must be resolved before use; no citation is invented here.

| ID | Supplied title/name | Status | Required resolution |
|---|---|---|---|
| S014 | Quantized Spike-Driven Transformer | BLOCKED | primary paper, version and official code |
| S015 | Efficient Sparse Hardware Accelerator for Spike-Driven Transformer | BLOCKED | primary paper and hardware target |
| S016 | SimST | BLOCKED | disambiguated title, primary paper and code |
| S017 | SpikeVision | BLOCKED | disambiguated title, primary paper and code |
| S018 | Spike-Driven Transformer / Spikingformer | BLOCKED | exact papers/versions relevant to accounting |

## Repository-local sources already inspected

| ID | Source | Status | Verified use |
|---|---|---|---|
| L001 | `src/etsr/`, `tests/`, configs and shell scripts at `806c0aa` | VERIFIED | actual frame-first contract, model/state/profiler behavior |
| L002 | Git history through `806c0aa` | VERIFIED | branch ancestry and deleted smoke artifacts |
| L003 | archived DVS-GC documentation | VERIFIED AS HISTORICAL PROSE | prior intent/results context, not current implementation authority |
| L004 | charter revision v2 | VERIFIED AS OWNER BRIEF | scientific mandate and work order, not independent literature proof |

## Review record template

For each S-source, append a dated block after reading:

```text
Source ID and exact version:
Date read:
Paper sections/tables:
Supplement:
Official code commit/tag:
FACTS extracted:
What we use:
What we do NOT claim:
Implementation semantics:
Training/split caveats:
Parameter/state/cost caveats:
Open questions:
Matrix/Pareto cells updated:
```

The review is complete only when claim boundaries, comparison caveats and exact code version are
recorded—not when a link has merely been opened.

## S001 review — MSTP / original DVS-Lip

**Source ID and exact version:** CVPR 2022 paper pp. 20094–20103; official supplement; repository
commit `a0246506f4d4f3d15f187b18ae9c1010aa146cdb` (latest public `main` reviewed 2026-08-21).

**Paper sections/tables:** §3.1–3.2, §4.1–4.3, Tables 1–3 and Fig. 5.

**Supplement:** vocabulary selection and exact Part 1/Part 2 word lists.

**Official code:** `utils/dataset.py`, `utils/utils.py`, `model/model.py`, `main.py`, README, license
and `data/frame_nums.json`.

**FACTS extracted:** DVS-Lip has 100 words, 40 speakers and 19,871 samples; official partitions are
14,896/30 speakers and 4,975/10 disjoint speakers. MSTP converts `(t,x,y,p)` events to one signed,
temporally interpolated voxel channel at low/high rates, uses dual ResNet-18 branches plus message
flow, then a 3-layer bidirectional GRU and temporal mean. Paper MSTP at `(30,210)` reports overall
72.10%, Acc1 62.17% and Acc2 82.07%. Acc1 is paper Part 1 (confusable pairs), Acc2 Part 2 (common).

**What we use:** official counts; raw-event field expectations pending sample inspection; semantic
class groups; multi-rate representation as a prior-art baseline; published accuracy as a literature
anchor only.

**What we do NOT claim:** speaker IDs, a valid 24/6 split, dataset licensing, causal/streaming
operation, reproduced accuracy, parameter/state/energy cost, or that MSTP's fixed voxel encoding is
canonical for this project.

**Implementation semantics:** repository layout is `train|test/<word>/<integer>.npy`.
`frame_nums.json` totals match the paper counts but carries per-sample frame counts, not speakers.
The paper configuration has 30/210 bins; the current README test command uses `1+7` while its train
command uses `1+4`. The model backend is bidirectional and non-causal.

**Training/split caveats:** the official training loop evaluates `test/` every epoch and selects the
best checkpoint from it. No validation partition or speaker manifest is supplied. The code function
named `compute_each_part_acc` assigns the supplement's Part 1 words to its `acc_part2` accumulator,
apparently reversing the paper's Acc1/Acc2 semantics.

**Parameter/state/cost caveats:** S001 does not provide a complete persistent-state or hardware-cost
account. A local instantiation of the public code yields configuration-dependent counts that do not
replace a paper-reported value; no Pareto conclusion is drawn from them.

**Open questions:** dataset-specific terms; exact archive/dtype/time origin; sample-to-speaker map;
exact training recipe that produced the 30/210 result; reason for the public Acc1/Acc2 reversal.

**Matrix/Pareto cells updated:** N01 and MSTP row. Full protocol findings are in
[`DVSLIP_PROTOCOL.md`](DVSLIP_PROTOCOL.md).

## S003 review — SpikGRU2+

**Source ID and exact version:** CVPRW Embedded Vision Workshop 2024 paper pp. 2141–2151;
repository commit `a61c6afcdf5f0398646a885ce416da690cf8351f` (latest public `main` reviewed
2026-08-21).

**Paper sections/tables:** §2.1–2.3, §3.1–3.6, §4.1–4.3, Tables 1–6.

**Supplement:** none linked/required by the official repository.

**Official code:** `DVSLip.py`, `layers.py`, `models.py`, `main.py`, README and license.

**FACTS extracted:** simulation uses 90 nearest-neighbor time bins over 1.2 s, separate positive and
negative channels and 88×88 crops. The SNN combines a spiking ResNet-18 frontend with learned
per-channel PLIF decays and a 3-layer bidirectional two-gate SpikGRU2+ backend using signed spikes.
The paper reports 58.6M parameters and 75.3% overall accuracy after sparsity fine-tuning. At that
point it reports 0.124 operations/synapse, 3.7 GOP/s and an estimated 1.3–11× energy reduction range
relative to its ANN model assumptions.

**What we use:** strong recurrence/augmentation anchor; evidence that temporal masking and core
choice materially affect DVS-Lip; explicit non-causal/full-scale comparison point.

**What we do NOT claim:** gating alone caused the gain; the result is causal, streaming, compact or
FPGA-measured; 58.6M demonstrates sub-500k feasibility; the estimated energy range transfers to our
hardware contract.

**Implementation semantics:** positive/negative event counts are accumulated separately; timestamps
are rounded using a fixed 40 ms base divided by `T/30` and events beyond `T` are dropped. Membrane
and recurrent states are initialized inside sequence forwards. The default model is bidirectional;
the public code has an optional unidirectional flag, but no reported primary result or streaming
equivalence for it.

**Training/split caveats:** SNN training is 100 epochs at fixed 3e-4 plus 100 epochs cosine
fine-tuning, with warmup, weight decay, dropout and strong spatial/temporal augmentation. The code
passes the official test loader as `valid_dataloader`, evaluates it every epoch and selects the best
checkpoint. It supplies no speaker IDs or validation manifest.

**Parameter/state/cost caveats:** paper parameters are dominated by the full-width bidirectional
backend (47M backend versus 11M frontend in its discussion). Energy is model-based, not a measured
deployment. Full-precision neuronal state and memory traffic are not reduced to this project's
state-bit/traffic contract.

**Open questions:** none of S003's code resolves the DVS-Lip speaker mapping or dataset license;
causal accuracy and state cost remain unreported for the target constraints.

**Matrix/Pareto cells updated:** N06 and SpikGRU2+ row.

## S006 review — Neuromorphic Sequential Arena

**Source ID and exact version:** IJCAI 2025 paper *Neuromorphic Sequential Arena: A Benchmark for
Neuromorphic Temporal Processing* and official supplement; repository commit
`af3c3204df0c3bedc9abfde3a01cc62f42effde5` (latest public `main` reviewed 2026-08-21).

**Paper sections/tables:** §3.1–3.3, §4, Tables 2–5; supplement dataset preprocessing,
task-specific configurations and Tables S5–S6.

**Official code:** `experiments/neuromorphic_sequential_arena/ALR/{main.py,run_all.sh}`,
`src/neuroseqbench/datasets/dvs_lip.py`, relevant neuron/network implementations, Git history and
GPL-3.0 license. The reviewed post-publication commit only passes the already-created `device` into
the training function; it does not change the published ALR data/model recipe.

**FACTS extracted:** the ALR task converts DVS-Lip events to 200 nearest temporal bins, preserves
polarity in two channels, crops to 88×88 and uses final-step classification. In the paper's
approximately 9.5M-parameter comparison, SFNN results are LIF 17.83%, CE-LIF 48.32%, LTC 48.93%,
SPSN 45.73% and PMSN 57.43%; SRNN results are LIF 34.71%, CE-LIF 51.64% and LTC 56.64%.
Architecture results with LIF are GSN 21.17%, Spiking TCN 47.14%, Spike-Driven Transformer 39.62%,
Binary S4D 44.80% and GSU 41.35%.

**What we use:** evidence that temporal neuron/core selection materially changes ALR performance;
PMSN-SFNN, LTC-SRNN, CE-LIF-SRNN, causal Spiking TCN and the non-causal Transformer as distinct
comparison anchors; `T=200` two-polarity frames as another published encoding variant.

**What we do NOT claim:** these results establish sub-500k feasibility; the Transformer result
rejects attention; arithmetic energy estimates are DVS-Lip hardware measurements; any method is
streaming merely because its offline computation is causal.

**Implementation semantics:** the ALR loader eagerly materializes all `.npy` samples and exposes no
speaker identity. FFSNN/SRNN flatten every 88×88×2 timestep before six dense layers. Serial
CE-LIF/LTC neurons support state arguments internally, but the ALR model has no first-class
step/chunk interface. PMSN computes a causal sequence kernel with an FFT-based full-sequence path;
its public state path is not evidence of chunk equivalence. Spiking TCN uses causal temporal
convolution. The Spike-Driven Transformer uses unmasked attention over the event-bin sequence and
is therefore non-causal in this configuration.

**Training/split caveats:** the code names the official `test/` dataset `val_dataset`, evaluates it
each epoch and chooses the best result/checkpoint. No validation or speaker manifest is supplied.
Experiments use 100 epochs, batch 256, AdamW/cross-entropy and the model-specific settings in
`run_all.sh`. The supplement lists LTC-SFNN as `512*6`, while the result-bearing official command
uses `352*6`; the published 48.93% row is retained as an accuracy anchor, not an unambiguous
reproduction recipe.

**Parameter/state/cost caveats:** the supplement states approximately 9.5M trainable parameters for
ALR fairness, over nineteen times the preferred 500k target. Table 5 efficiency results concern the
synthetic adding task, not ALR. Its energy proxy uses assumed 45 nm MAC/AC costs and does not include
a measured DVS-Lip deployment or this project's complete state/memory-traffic accounting.

**Open questions:** exact per-configuration counts rather than the approximate fairness target;
why the LTC-SFNN width differs; causal/chunk accuracy under a test-embargoed protocol. S006 does not
resolve dataset terms, actual archive semantics or sample-to-speaker mapping.

**Matrix/Pareto cells updated:** N08a–N08f and six NSA rows.
