# Novelty matrix

**Status:** structure complete; literature cells pending P1-01 primary-source review
**Rule:** a `TODO` cell carries no novelty claim.

To keep the matrix reviewable, required columns are split into two panels joined by `ID`. Both panels
must be completed before P1-02 can be marked done.

## Review tracker

| ID | Method | Source ID | Review status |
|---|---|---|---|
| N01 | MSTP | S001 | LISTED |
| N02 | TVTA | S002 | LISTED |
| N03 | Spiking Patches | S007 | LISTED |
| N04 | Event2Vec | S005 | LISTED |
| N05 | FATE | S008 | LISTED |
| N06 | SpikGRU2+ | S003 | LISTED |
| N07 | Mul-free Channel-wise PSN | S004 | LISTED |
| N08 | NSA PMSN/LTC/CE-LIF baselines | S006 | LISTED |
| N09 | RVT | S010 | LISTED |
| N10 | TEFormer | S011 | LISTED |
| N11 | BAM-SLDK | S012 | LISTED |
| N12 | Current Mini-QKFormer | L001 | CODE VERIFIED |
| N13 | Proposed representation/core candidate | future decision | NOT SELECTED |

## Panel A — representation and architecture

| ID | Input primitive | Temporal discretization | Spatial architecture | Temporal architecture | Temporal state | State update | Readout | Causal? | Bidirectional? | Asynchronous? | Learned temporal parameters? | Temporal compression location | Spatial compression location |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N01–N11 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| N12 | dense count frames | fixed `T` before model | hierarchical conv + QKTA + SSA | repeated LIF inside spatial stack | membrane local to one sequence forward | first-order LIF soft reset | temporal mean + linear head | yes within supplied sequence | no | no | LIF `tau` configured, not learned | dataset/preprocessing before model; final temporal mean | 128→16 initial embedding, then 16→8 stage |
| N13 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | required yes | required no | TODO | TODO | TODO | TODO |

## Panel B — training, cost, evidence and overlap

| ID | Training objective | Parameter count/storage | Persistent state | Main arithmetic | Memory behavior | Reported task | Reported accuracy | Streaming? | Code | FPGA/neuromorphic relevance | Direct overlap | Remaining differentiation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N01–N11 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | see `SOURCES.md` | TODO | TODO | TODO |
| N12 | supervised cross-entropy | configuration-dependent; count at runtime | no cross-call persistent state; intra-forward LIF membrane not profiled in bits | dense first/head MAC, activity-weighted AC, spike comparisons/reset | state traffic not counted | DVS-GC diagnostic | repository artifacts not present in this workspace | no first-class step/chunk API | local `src/etsr/models/` | arithmetic proxy only | baseline, not a novelty claim | candidate spatial mixer plus future explicit causal temporal state |
| N13 | TODO | preferred target ~500k, not proven | must be quantified | TODO | must be quantified | DVS-Lip then DailyDVS-200 | TODO | required yes | TODO | causal, quantizable, state-minimal | cannot assess before review | must be specific beyond generic recurrence/attention |

## Completion rule

Replace grouped rows `N01–N11` with one row per method after its source is verified. Record paper
table/section, code commit and caveats in `SOURCES.md`. A final novelty claim requires completed
`Direct overlap` and `Remaining differentiation` cells, not merely a new module name.
