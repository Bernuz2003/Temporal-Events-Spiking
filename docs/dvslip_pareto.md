# DVS-Lip Pareto table

**Status:** three source families verified; remaining numerical literature review pending P1-01
**Do not use this file to set success thresholds yet.**

The charter contains candidate headline values, but they are not copied into the verified table
until paper, supplement and relevant code/config are checked. This prevents a secondary brief from
becoming the citation source.

## Verified table

| Method | Year/venue | ANN/SNN | Representation | Spatial architecture | Temporal architecture | Causal? | Bidirectional? | Streaming? | Accuracy | Acc1 | Acc2 | Parameters or storage | Timesteps/resolution | Training recipe | SOP/FLOP | Energy | Persistent state | Official code | Comparability caveats | Evidence status |
|---|---|---|---|---|---|---|---|---|---:|---:|---:|---|---|---|---|---|---|---|---|---|
| MSTP | 2022/CVPR | ANN | one-channel signed interpolated voxel grids at two rates | dual 3D ResNet-18 + message flow | 3-layer BiGRU + temporal mean | no | yes | no | 72.10 | 62.17 | 82.07 | not reported in S001; 60.3M re-reported by S003 | `(30,210)`, 88×88 | Adam, 80 epochs, cosine 3e-4→5e-6, crop/flip | not reported | not reported | BiGRU state, precision/bits not reported | S001 commit `a024650` | Acc1=confusable Part 1, Acc2=common Part 2; public metric code appears reversed; official test selected every epoch; README train command differs from paper bins | VERIFIED with protocol caveats |
| Spiking MSTP | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | — | — | — | TODO | TODO | TODO | TODO | TODO | TODO | resolve in S001 | TODO | LISTED |
| SpikGRU2+ | 2024/CVPRW-EVW | SNN | nearest-bin two-polarity event frames | spiking ResNet-18 with learned PLIF decay | 3-layer bidirectional two-gate signed SpikGRU2+ | no | yes | no implementation evidence | 75.30 | — | — | 58.6M | 90 bins/1.2 s (~13 ms), 88×88 | batch 32; 100 epochs fixed LR + 100 cosine fine-tuning; warmup, spatial/temporal augmentation; spike-loss fine-tune | 3.7 GOP/s and 0.124 OP/syn at 75.3% | estimated 1.3–11× reduction vs ANN assumptions; not measured | PLIF membrane + bidirectional recurrent state, full precision/unquantified bits | S003 commit `a61c6af` | official test used as validation/checkpoint selector; no speaker metadata; non-causal and >100× preferred parameter target | VERIFIED with protocol caveats |
| Mul-free Channel-wise PSN model | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | — | — | — | TODO | TODO | TODO | TODO | TODO | TODO | S004 | TODO | LISTED |
| Event2Vec random events | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | — | — | — | TODO | TODO | TODO | TODO | TODO | TODO | S005 | TODO | LISTED |
| Event2Vec clustered events | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | — | — | — | TODO | TODO | TODO | TODO | TODO | TODO | S005 | TODO | LISTED |
| TVTA | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | — | — | — | TODO | TODO | TODO | TODO | TODO | TODO | S002 | TODO | LISTED |
| NSA LIF-SFNN | 2025/IJCAI | SNN | nearest-bin two-polarity frames | flattened 88×88×2 input + six 512-wide dense layers | feedforward LIF dynamics + final step | yes | no | not demonstrated | 17.83 | — | — | ≈9.5M matched regime | 200 bins, 88×88 | batch 256, 100 epochs, AdamW/CE, triangle surrogate | not reported for ALR | not reported for ALR | intra-sequence LIF membrane; bits not reported | S006 commit `af3c320` | official test selected each epoch; large flattened input projection; ordinary-LIF control, not target-scale result | VERIFIED with protocol caveats |
| NSA PMSN-SFNN | 2025/IJCAI | SNN | nearest-bin two-polarity frames | flattened input + six 512-wide dense layers | causal PMSN sequence kernel + final step | yes by offline graph | no | not demonstrated | 57.43 | — | — | ≈9.5M matched regime | 200 bins, 88×88 | batch 256, 100 epochs, AdamW/CE, triangle surrogate | not reported for ALR | not reported for ALR | learned linear-system history; chunk state not verified | S006 commit `af3c320` | official test selected each epoch; full-sequence FFT path; no speaker manifest | VERIFIED with protocol caveats |
| NSA CE-LIF-SRNN | 2025/IJCAI | SNN | nearest-bin two-polarity frames | flattened input + six 460-wide recurrent dense layers | recurrent contextual-embedding LIF + final step | yes | no | not demonstrated | 51.64 | — | — | ≈9.5M matched regime | 200 bins, 88×88 | batch 256, 100 epochs, AdamW/CE, triangle surrogate | not reported for ALR | not reported for ALR | membrane, spike, adaptive threshold and recurrent state; bits not reported | S006 commit `af3c320` | official test selected each epoch; no first-class step/chunk API | VERIFIED with protocol caveats |
| NSA LTC-SRNN | 2025/IJCAI | SNN | nearest-bin two-polarity frames | flattened input + six 340-wide recurrent dense layers | recurrent learned time constants + final step | yes | no | not demonstrated | 56.64 | — | — | ≈9.5M matched regime | 200 bins, 88×88 | batch 256, 100 epochs, AdamW/CE, triangle surrogate | not reported for ALR | not reported for ALR | membrane, spike, adaptive threshold and recurrent state; bits not reported | S006 commit `af3c320` | official test selected each epoch; supplement/code width discrepancy affects LTC-SFNN, not this row | VERIFIED with protocol caveats |
| NSA Spiking TCN | 2025/IJCAI | SNN | nearest-bin two-polarity frames | flattened input projected to six 75-channel layers | causal dilated TCN, kernel 7, + final step | yes | no | chunk equivalence not reported | 47.14 | — | — | ≈9.5M matched regime | 200 bins, 88×88 | batch 256, 100 epochs, AdamW/CE, triangle surrogate | not reported for ALR | not reported for ALR | finite convolution buffers and LIF state; bits not reported | S006 commit `af3c320` | official test selected each epoch; required causal non-recurrent anchor | VERIFIED with protocol caveats |
| NSA Spike-Driven Transformer | 2025/IJCAI | SNN | nearest-bin two-polarity frames | flattened input projected to six 272-wide layers | unmasked four-head spike-driven self-attention + final step | no | no | no | 39.62 | — | — | ≈9.5M matched regime | 200 bins, 88×88 | batch 256, 100 epochs, AdamW/CE, triangle surrogate | not reported for ALR | not reported for ALR | full-sequence activations and LIF state; bits not reported | S006 commit `af3c320` | `D=1`, moderate dataset/scale caveat; does not establish Transformer unsuitability | VERIFIED with protocol caveats |
| Relevant current event-token model | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | — | — | — | TODO | TODO | TODO | TODO | TODO | TODO | unresolved | selection pending | PENDING |

## Comparison rules

- Separate comparable rows for pretraining/fine-tuning, augmentation and representation variants.
- Never convert MB to parameter count without verified precision/storage convention.
- Separate causal from bidirectional/non-causal models.
- Do not infer streaming from causality alone.
- Report missing state/energy as `not reported`, not zero.
- Record official versus reconstructed Acc1/Acc2 definitions.
- A paper at ResNet-18 or tens-of-millions scale cannot prove sub-500k feasibility.
- Hardware proxies and measured hardware results occupy different fields/notes.

## Threshold gate

Success thresholds may be proposed only after required anchor rows are verified and the 0.5M/1M/2M
capacity scan is available. Any threshold change is a decision-record event.
