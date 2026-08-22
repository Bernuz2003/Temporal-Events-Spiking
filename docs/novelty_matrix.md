# Novelty matrix

**Status:** structure complete; three primary-source families verified, remaining cells pending P1-01
**Rule:** a `TODO` cell carries no novelty claim.

To keep the matrix reviewable, required columns are split into two panels joined by `ID`. Both panels
must be completed before P1-02 can be marked done.

## Review tracker

| ID | Method | Source ID | Review status |
|---|---|---|---|
| N01 | MSTP | S001 | VERIFIED |
| N02 | TVTA | S002 | LISTED |
| N03 | Spiking Patches | S007 | LISTED |
| N04 | Event2Vec | S005 | LISTED |
| N05 | FATE | S008 | LISTED |
| N06 | SpikGRU2+ | S003 | VERIFIED |
| N07 | Mul-free Channel-wise PSN | S004 | LISTED |
| N08a | NSA LIF-SFNN control | S006 | VERIFIED |
| N08b | NSA PMSN-SFNN | S006 | VERIFIED |
| N08c | NSA CE-LIF-SRNN | S006 | VERIFIED |
| N08d | NSA LTC-SRNN | S006 | VERIFIED |
| N08e | NSA Spiking TCN | S006 | VERIFIED |
| N08f | NSA Spike-Driven Transformer | S006 | VERIFIED |
| N09 | RVT | S010 | LISTED |
| N10 | TEFormer | S011 | LISTED |
| N11 | BAM-SLDK | S012 | LISTED |
| N12 | Current Mini-QKFormer | L001 | CODE VERIFIED |
| N13 | Proposed representation/core candidate | future decision | NOT SELECTED |

## Panel A — representation and architecture

| ID | Input primitive | Temporal discretization | Spatial architecture | Temporal architecture | Temporal state | State update | Readout | Causal? | Bidirectional? | Asynchronous? | Learned temporal parameters? | Temporal compression location | Spatial compression location |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N01 | raw `(t,x,y,p)` events | signed temporally interpolated voxel grids; paper `(30,210)` | dual 3D ResNet-18 branches + message flow | 3-layer BiGRU | GRU hidden state | standard learned GRU gates | temporal mean + linear/softmax | no | yes | no in public implementation | GRU weights; discretization fixed | high-rate branch reduced to low-rate sequence; final mean | 128→96→88 crop plus ResNet hierarchy |
| N02–N05 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| N06 | raw `(t,x,y,p)` events | 90 nearest bins over 1.2 s; positive/negative channels | spiking ResNet-18 | 3-layer bidirectional SpikGRU2+ | PLIF membrane plus recurrent current/output state | learned per-channel leak, two sigmoid gates, signed spike output | linear logits + temporal mean | no | yes | synchronous simulation; deployment only proposed | learned PLIF/GRU decays and gates | recurrent backend after spatial pooling | 128→88 crop plus ResNet hierarchy |
| N07 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| N08a | raw `(t,x,y,p)` events | 200 nearest bins; positive/negative channels | flattened 88×88×2 input + six dense layers | feedforward LIF sequence dynamics | LIF membrane | serial soft-reset LIF | final-step linear logits | yes | no | no | fixed LIF decay | final-step readout | no spatial hierarchy; immediate flattening |
| N08b | same as N08a | same as N08a | flattened input + six 512-wide dense layers | parallel multi-compartment spiking neuron (PMSN) | four-mode learned linear-system state represented as a sequence kernel | causal convolution evaluated by FFT; surrogate threshold | final-step linear logits | yes by offline graph | no | no | learned state-space kernel and skip | final-step readout | no spatial hierarchy; immediate flattening |
| N08c | same as N08a | same as N08a | flattened input + six 460-wide recurrent dense layers | recurrent CE-LIF | membrane, spike and adaptive threshold | serial recurrence plus time-indexed context embedding | final-step linear logits | yes | no | no | learned context embedding; fixed decay/threshold beta | final-step readout | no spatial hierarchy; immediate flattening |
| N08d | same as N08a | same as N08a | flattened input + six 340-wide recurrent dense layers | recurrent LTC | membrane, spike and adaptive threshold state | serial learned membrane/threshold time constants plus recurrence | final-step linear logits | yes | no | no | learned state-conditioned time constants and recurrent weights | final-step readout | no spatial hierarchy; immediate flattening |
| N08e | same as N08a | same as N08a | flattened input projected to six 75-channel layers | spiking causal TCN, kernel 7 | finite temporal convolution history plus LIF state | dilated causal convolution and LIF | final-step linear logits | yes | no | no | learned temporal kernels; fixed LIF dynamics | receptive-field hierarchy then final step | no spatial hierarchy; immediate flattening |
| N08f | same as N08a | same as N08a | flattened input projected to six 272-wide layers | unmasked Spike-Driven Transformer, four heads | full-sequence token activations plus LIF membrane | spike-driven self-attention over all bins | final-step linear logits | no | no | no | attention/FFN weights; fixed LIF dynamics | global attention before final step | no spatial hierarchy; immediate flattening |
| N09–N11 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| N12 | dense count frames | fixed `T` before model | hierarchical conv + QKTA + SSA | repeated LIF inside spatial stack | membrane local to one sequence forward | first-order LIF soft reset | temporal mean + linear head | yes within supplied sequence | no | no | LIF `tau` configured, not learned | dataset/preprocessing before model; final temporal mean | 128→16 initial embedding, then 16→8 stage |
| N13 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | required yes | required no | TODO | TODO | TODO | TODO |

## Panel B — training, cost, evidence and overlap

| ID | Training objective | Parameter count/storage | Persistent state | Main arithmetic | Memory behavior | Reported task | Reported accuracy | Streaming? | Code | FPGA/neuromorphic relevance | Direct overlap | Remaining differentiation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N01 | cross-entropy; Adam/cosine, 80 epochs | not reported by S001; later comparison reports 60.3M | bidirectional GRU state, not quantified | dense 3D CNN + GRU | not reported | DVS-Lip | 72.10%; Acc1 62.17%; Acc2 82.07% | no | official S001 commit recorded | no measured hardware result | multi-rate frame encoding plus long non-causal recurrence | compact causal raw-event representation/core with explicit state and fair scale controls |
| N02–N05 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | see `SOURCES.md` | TODO | TODO | TODO |
| N06 | cross-entropy; 100 fixed-LR + 100 cosine fine-tuning; spatial/temporal augmentation | 58.6M reported | full-precision PLIF and bidirectional recurrent state; bits not reported | spiking convolution/recurrent operations; 3.7 GOP/s reported at 75.3% | modeled operations/memory energy range; no deployment measurement | DVS-Lip | 75.3% overall; Acc1/Acc2 not reported | no | official S003 commit recorded | neuromorphic energy estimate, not FPGA measurement | signed two-gate spiking recurrence and temporal masking | causal unidirectional state, orders-of-magnitude smaller model, quantified state/traffic |
| N07 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | see `SOURCES.md` | TODO | TODO | TODO |
| N08a | AdamW/cross-entropy, 100 epochs | ≈9.5M comparison regime | intra-sequence LIF state; bits not reported | dense MAC + spike dynamics | full dense 15,488→512 input projection per bin | DVS-Lip/ALR | 17.83% | no public step/chunk API | S006 commit `af3c320` | control only; no ALR efficiency measurement | establishes weak ordinary-LIF control at this scale/encoding | target must distinguish core effect from flattened-input capacity and protocol leakage |
| N08b | AdamW/cross-entropy, 100 epochs | ≈9.5M comparison regime | linear-system history; exported state/chunk contract not verified | dense MAC + FFT sequence convolution | full sequence plus dense input projection | DVS-Lip/ALR | 57.43% | not demonstrated | S006 commit `af3c320` | parallel temporal computation, no ALR deployment measurement | learned long-kernel temporal state is direct prior art | compact causal state, online equivalence and event-native spatial compression |
| N08c | AdamW/cross-entropy, 100 epochs | ≈9.5M comparison regime | membrane/spike/threshold plus recurrence; bits not reported | dense MAC + recurrent/state updates | six dense recurrent layers; no traffic report | DVS-Lip/ALR | 51.64% | not demonstrated | S006 commit `af3c320` | no ALR efficiency measurement | contextual thresholds plus explicit recurrence are direct prior art | smaller explicit state and controlled separation from representation |
| N08d | AdamW/cross-entropy, 100 epochs | ≈9.5M comparison regime | membrane/spike/threshold plus recurrence; bits not reported | dense MAC + recurrent learned time-constant updates | six dense recurrent layers; no traffic report | DVS-Lip/ALR | 56.64% | not demonstrated | S006 commit `af3c320` | no ALR efficiency measurement | adaptive time constants plus recurrence are direct prior art | compact state/update and fair causal comparison under embargo |
| N08e | AdamW/cross-entropy, 100 epochs | ≈9.5M comparison regime | convolution history and LIF state; bits not reported | dense input MAC + dilated temporal convolution | finite history buffers; no traffic report | DVS-Lip/ALR | 47.14% | causal implementation, chunk equivalence not reported | S006 commit `af3c320` | no ALR efficiency measurement | causal convolution is a required non-recurrent temporal baseline | target must beat/cost-match it and quantify buffers |
| N08f | AdamW/cross-entropy, 100 epochs | ≈9.5M comparison regime | full-sequence attention activations and LIF state | dense MAC + spike-driven attention | global sequence attention; no traffic report | DVS-Lip/ALR | 39.62% | no | S006 commit `af3c320` | no ALR efficiency measurement | spiking attention at matched large scale is prior art, not a rejection result | causal/local attention or different spatial-temporal allocation must be isolated |
| N09–N11 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | see `SOURCES.md` | TODO | TODO | TODO |
| N12 | supervised cross-entropy | configuration-dependent; count at runtime | no cross-call persistent state; intra-forward LIF membrane not profiled in bits | dense first/head MAC, activity-weighted AC, spike comparisons/reset | state traffic not counted | DVS-GC diagnostic | repository artifacts not present in this workspace | no first-class step/chunk API | local `src/etsr/models/` | arithmetic proxy only | baseline, not a novelty claim | candidate spatial mixer plus future explicit causal temporal state |
| N13 | TODO | preferred target ~500k, not proven | must be quantified | TODO | must be quantified | DVS-Lip then DailyDVS-200 | TODO | required yes | TODO | causal, quantizable, state-minimal | cannot assess before review | must be specific beyond generic recurrence/attention |

## Completion rule

Replace grouped rows `N01–N11` with one row per method after its source is verified. Record paper
table/section, code commit and caveats in `SOURCES.md`. A final novelty claim requires completed
`Direct overlap` and `Remaining differentiation` cells, not merely a new module name.
