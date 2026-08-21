# DVS-Lip Pareto table

**Status:** required schema and anchor list prepared; numerical literature review pending P1-01
**Do not use this file to set success thresholds yet.**

The charter contains candidate headline values, but they are not copied into the verified table
until paper, supplement and relevant code/config are checked. This prevents a secondary brief from
becoming the citation source.

## Verified table

| Method | Year/venue | ANN/SNN | Representation | Spatial architecture | Temporal architecture | Causal? | Bidirectional? | Streaming? | Accuracy | Acc1 | Acc2 | Parameters or storage | Timesteps/resolution | Training recipe | SOP/FLOP | Energy | Persistent state | Official code | Comparability caveats | Evidence status |
|---|---|---|---|---|---|---|---|---|---:|---:|---:|---|---|---|---|---|---|---|---|---|
| MSTP | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | — | — | — | TODO | TODO | TODO | TODO | TODO | TODO | S001 | TODO | LISTED |
| Spiking MSTP | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | — | — | — | TODO | TODO | TODO | TODO | TODO | TODO | resolve in S001 | TODO | LISTED |
| SpikGRU2+ | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | — | — | — | TODO | TODO | TODO | TODO | TODO | TODO | S003 | TODO | LISTED |
| Mul-free Channel-wise PSN model | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | — | — | — | TODO | TODO | TODO | TODO | TODO | TODO | S004 | TODO | LISTED |
| Event2Vec random events | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | — | — | — | TODO | TODO | TODO | TODO | TODO | TODO | S005 | TODO | LISTED |
| Event2Vec clustered events | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | — | — | — | TODO | TODO | TODO | TODO | TODO | TODO | S005 | TODO | LISTED |
| TVTA | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | — | — | — | TODO | TODO | TODO | TODO | TODO | TODO | S002 | TODO | LISTED |
| NSA neuron/architecture baselines | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | — | — | — | TODO | TODO | TODO | TODO | TODO | TODO | S006 | separate rows required | LISTED |
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
