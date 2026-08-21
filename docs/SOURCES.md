# Source register

**Purpose:** track what has actually been read and which claims are allowed.
**Current state:** bibliography imported from charter v2; primary-source deep review is P1-01 and has
not been completed in this iteration.

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
| S001 | MSTP / original DVS-Lip, Tan et al., CVPR 2022 | <https://openaccess.thecvf.com/content/CVPR2022/papers/Tan_Multi-Grained_Spatio-Temporal_Features_Perceived_Network_for_Event-Based_Lip-Reading_CVPR_2022_paper.pdf> | <https://github.com/tgc1997/event-based-lip-reading> | LISTED | dataset/protocol, Acc1/Acc2, multi-rate representation | exact split/layout or reproduced accuracy |
| S002 | TVTA | <https://arxiv.org/abs/2607.08236> | not supplied | LISTED | temporal modeling before spatial aggregation; novelty boundary | architecture details or comparable cost |
| S003 | SpikGRU2+ | <https://openaccess.thecvf.com/content/CVPR2024W/EVW/papers/Dampfhoffer_Neuromorphic_Lip-Reading_with_Signed_Spiking_Gated_Recurrent_Units_CVPRW_2024_paper.pdf> | <https://github.com/manondampfhoffer/SpikGRU-DVSLip> | LISTED | accuracy anchor, recurrence, augmentation, causality/scale caveats | gating-only attribution or compact feasibility |
| S004 | Multiplication-Free Channel-wise PSN | <https://papers.nips.cc/paper_files/paper/2025/file/5f40a99efedc1148a061d80201949c84-Paper-Conference.pdf> | not supplied | LISTED | shift/add temporal dynamics and DVS-Lip anchor | sub-500k or FPGA result |
| S005 | Event2Vec | <https://arxiv.org/abs/2504.15371> | <https://github.com/Intelligent-Computing-Lab-Panda/event2vec> | LISTED | event-native representation, long-memory behavior, recipe | exact parameter count inferred from storage |
| S006 | Neuromorphic Sequential Arena | <https://www.ijcai.org/proceedings/2025/0544.pdf> | <https://github.com/liyc5929/neuroseqbench> | LISTED | core/neuron sensitivity and trade-offs | sub-500k Pareto proof or Transformer rejection |
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
