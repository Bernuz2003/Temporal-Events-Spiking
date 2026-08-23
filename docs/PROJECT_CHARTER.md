# Temporal Event Spiking Research — Project Charter & Operating Manual for the AI Research Engineer

**Status:** active parent document / authoritative implementation brief
**Revision:** v2 — architecture/temporal-core alignment after senior review
**Revision date:** 2026-08-21
**Target phase:** transition from DVS-Gesture-Chain audit to DVS-Lip architecture construction and hardware-aware temporal modeling
**Repository snapshot reviewed:** `refactor/mechanistic-temporal-audit`
**Reviewed HEAD:** `806c0aa`
**`master` at snapshot:** `02cd224`
**Active working branch:** `developer`, created from `806c0aa` after tagging the completed audit state as `dvsgc-audit-complete-2026`
**Audience:** autonomous AI research/implementation agent acting as a senior research engineer and co-owner of the project
**Primary scientific owner:** thesis project, Politecnico di Torino / SMILIES workflow
**Language convention:** code, identifiers and commit messages in English; project notes may be in Italian or English, but scientific terminology must remain precise.

Operational navigation and mutable project state live in [`docs/README.md`](README.md). This
charter remains the stable scientific constitution; day-to-day status, decisions and verification
evidence must be updated in the child documents defined in section 36.

---

# 0. Why this document exists

This file is the **parent document** of the repository.

Its job is to preserve, across agents, experiments, branches and changes in the literature:

- the scientific objective;
- the engineering objective;
- the evidence already accumulated;
- the assumptions that are still hypotheses rather than facts;
- the decision process;
- the experimental gates;
- the hardware constraints;
- the reproducibility rules;
- the definition of a thesis-worthy result;
- the exact boundaries between required work and stretch work.

This document must be treated as the project constitution.

It is **not** a static implementation recipe. If evidence changes the scientific strategy, this file must be updated through an explicit decision record. It must never silently drift away from the actual implementation.

The implementing agent must behave as a **senior research engineer who owns the scientific quality of the project**, not as a blind executor.

The agent is expected to:

- reconstruct context before editing code;
- challenge assumptions when evidence is weak;
- distinguish facts, inferences, hypotheses, design preferences and decisions;
- inspect primary papers and official repositories whenever implementation details matter;
- propose simpler or stronger alternatives when appropriate;
- identify confounds before spending GPU time;
- refuse to hide uncertainty behind implementation activity;
- preserve negative results and abandoned hypotheses;
- maintain documentation as part of implementation;
- optimize for the real objective: a compact, accurate, causal, transferable and FPGA-plausible temporal event model.

If a requested implementation conflicts with evidence, reproducibility, experimental validity, test embargo, parameter/state budget or hardware feasibility, the agent must **stop, explain the conflict and propose an alternative before implementing**.

---

# 1. Executive decision — project direction

## 1.1 Continue the existing repository

Do **not** start from a clean repository.

Continue from:

```text
refactor/mechanistic-temporal-audit @ 806c0aa
```

and create the active branch:

```text
developer
```

after tagging the completed DVS-GC phase.

Recommended transition:

```bash
git checkout refactor/mechanistic-temporal-audit
git tag -a dvsgc-audit-complete-2026 -m "Completed DVS-GC temporal/mechanistic audit phase"
git checkout -b developer
```

If a remote `developer` branch already exists in the SMILIES GitLab workflow, reconcile it explicitly. Never overwrite it blindly.

Do not develop directly on `master`.

## 1.2 Why not return to `master`

The refactor branch is a descendant of `master` and retains useful infrastructure that would otherwise have to be rebuilt:

- multi-seed experiment support;
- split and dataset manifests;
- hashing/provenance;
- prefix evaluation;
- richer metric handling;
- tracing utilities;
- operation/firing profiling;
- stronger validation logic;
- audit artifacts;
- tests added after `master`.

The mechanistic-audit additions are largely isolated enough that they can be frozen rather than deleted.

Starting from `master` would discard useful safeguards.

Starting from a new project would discard provenance, tested code, profiling infrastructure, experiment conventions and lessons already paid for with research time.

## 1.3 Critical qualification

The current repository is **not** ready to receive DVS-Lip by simply adding another dataset class.

The old data/training path is frame-first and the current model was built as a diagnostic Mini-QKFormer for the temporal-audit phase.

The correct transition is:

> keep the repository and history, but introduce a clean raw-event → representation → spatial core → temporal core → readout research layer.

Do not force DVS-Lip into DVS-GC abstractions.

---

# 2. Verified repository state

The following observations were verified on the reviewed snapshot.

## 2.1 Positive state

The test suite passed during review:

```text
30 passed
```

The following checks also completed successfully in the review runtime:

- Python compilation for `src/` and `tests/`;
- shell syntax check for existing scripts;
- `git diff --check`.

`ruff` was not available in that runtime and must be rerun in the project Singularity container.

Reusable package structure:

```text
src/etsr/
├── data/
├── evaluation/
├── models/
├── profiling/
├── training/
├── utils/
├── cli.py
├── config.py
├── reproducibility.py
└── runner.py
```

Reusable components include:

- `MiniQKFormer`;
- explicit `MultiStepLIF`;
- MAC/AC operation profiling;
- firing-rate profiling;
- Horowitz energy proxy;
- checkpoint selection;
- run artifacts;
- Git commit and dirty-state logging;
- dataset/split manifest hashing;
- prefix metrics;
- tracing/evaluation infrastructure.

## 2.2 Repository/documentation drift to fix in P0

Known inconsistencies include:

- `README.md` references `configs/smoke.yaml`, but the branch deleted it;
- `Makefile` still references `scripts/smoke_test.sh`, which was deleted;
- `docs/repository_tree.md` still lists deleted smoke files;
- `docs/validation.md` is stale relative to the current test suite;
- the ZIP contains untracked notebooks whose intended lifecycle is unclear.

These are reproducibility defects, not cosmetic details.

An autonomous agent relies on repository documentation. Contradictory documentation can cause incorrect implementation decisions.

## 2.3 Current architectural/software limitations

### Frame-first data contract

The old path expects samples standardized as:

```text
[T, C, H, W]
```

This cannot be the canonical DVS-Lip representation.

Raw DVS-Lip samples must retain:

```text
(x, y, t, polarity)
```

until an explicit encoder/representation module is applied.

### Dense-frame training contract

The current loop assumes:

```python
for frames, targets, indices in loader:
    logits = model(frames)
```

This couples dataset, temporal representation and model input.

The new phase requires:

```text
raw event sample
    ↓
temporal representation / encoder
    ↓
spatial feature extractor / mixer
    ↓
temporal core
    ↓
readout
```

### Streaming is not yet a first-class API

`MultiStepLIF` propagates state inside a sequence forward call, but state ownership is not yet exposed as a generic streaming contract.

The repository still needs explicit semantics for:

```text
reset_state
step
forward_sequence
detach_state
chunked inference
```

Do not claim full streaming deployment until offline/chunked/step execution equivalence is tested.

### Current profiler is incomplete for new stateful modules

The profiler handles conventional layers and current QK/SSA costs but will not automatically account for:

- state reads/writes;
- circular buffers;
- timestamp basis LUTs;
- delay-index reads;
- decay state updates;
- gate arithmetic;
- comparisons;
- bit shifts;
- stateful readout;
- recurrent state traffic.

Every new temporal component needs an explicit profiling contract and hardware card.

### Test/holdout semantics are too implicit

Some code infers test semantics from dataset names or artifact names.

For DVS-Lip, test embargo must be explicit in split metadata and enforced by code.

Never infer official-test use from a dataset class name.

---

# 3. Scientific objective

## 3.1 Final objective

The thesis is not trying to build a universal theory of SNN temporal cognition.

The target is:

> **Design a compact causal spiking architecture that achieves strong DVS-Lip performance by preserving useful event-time structure and exploiting it with an explicit but hardware-efficient temporal state, while remaining transferable to DailyDVS-200 and plausible for FPGA deployment.**

The desired system should move a practical Pareto frontier across:

\[
\text{Accuracy},
\quad
\text{Parameters},
\quad
\text{SOP},
\quad
E_{\mathrm{Horowitz}},
\quad
\text{Latency},
\quad
\text{Persistent State},
\quad
\text{Memory Traffic}.
\]

Accuracy is important and must be competitive for the model scale.

Accuracy alone is **not** the objective.

A model that gains accuracy by becoming tens of millions of parameters, non-causal or memory-heavy is outside the intended contribution.

## 3.2 Primary benchmark

**DVS-Lip** is the primary benchmark.

It is selected because:

- the task depends on continuous articulatory dynamics;
- visually confusable word pairs stress temporal discrimination;
- the official train/test protocol is speaker-disjoint;
- recent literature explicitly identifies insufficient temporal modeling before/around compression as a problem;
- the dataset is large enough for meaningful evaluation but still practical for a thesis;
- it exposes the tension between temporal fidelity, compute, memory and early recognition.

DVS-Lip is a **stress test** of temporal representation and memory.

It is not proof of universal event understanding.

## 3.3 External validation benchmark

**DailyDVS-200** is the planned visual transfer benchmark.

It is not a second architecture-search playground.

The purpose is to test whether the mechanism learned/designed on DVS-Lip transfers under minimal temporal calibration.

## 3.4 DVS-Gesture-Chain status

DVS-Gesture-Chain is **closed**.

Accepted minimal conclusion:

> Mini-QKFormer was not temporally blind, but DVS-GC order-2 was too small and too directly solvable from temporally resolved input statistics to serve as a strong benchmark for semantically rich temporal representation.

No new DVS-GC metric, audit, gate or architecture experiment is allowed unless the scientific owner explicitly reopens the phase.

DVS-GC remains:

- historical evidence;
- regression material;
- a record of methodological lessons.

It is not an active scientific target.

---

# 4. Critical revision introduced by this charter version

The most important correction relative to the previous charter is:

> **Mini-QKFormer is no longer assumed to be the full final backbone.**

It remains a valuable, compact and already-understood **candidate spatial backbone / baseline**, but its temporal dynamics must be treated as a hypothesis.

The previous audit asked:

> how does a small Spiking Transformer represent time?

The new construction phase asks:

> what compact architecture best solves a temporally demanding event task?

These are not the same question.

A diagnostic backbone must not become the final architectural prior by inertia.

---

# 5. Evidence that motivates an explicit temporal-core design

The literature does **not** support the simplistic conclusion “GRU good, Transformer bad”.

It supports a more precise conclusion:

> **DVS-Lip strongly rewards models that retain and exploit long temporal context; weak per-timestep processing plus ordinary LIF dynamics can be insufficient.**

The agent must keep the following evidence distinctions explicit.

## 5.1 Neuromorphic Sequential Arena (NSA)

Primary source:

https://www.ijcai.org/proceedings/2025/0544.pdf

On NSA’s ALR/DVS-Lip setting:

- LIF-SFNN: 17.83%;
- CE-LIF-SFNN: 48.32%;
- LTC-SFNN: 48.93%;
- PMSN-SFNN: 57.43%;
- LIF-SRNN: 34.71%;
- CE-LIF-SRNN: 51.64%;
- LTC-SRNN: 56.64%;
- Spike-Driven Transformer with LIF: 39.62%.

These numbers demonstrate strong sensitivity to temporal neuron/architecture choice.

They do **not** prove that 57–60% is achievable below 500k parameters.

NSA states that models are configured with comparable trainable parameter counts **within each task**, but the published table is not a sub-500k Pareto proof.

NSA also explicitly notes that Spike-Driven Transformer is evaluated with `D=1` for fairness and may be constrained by small model scale and moderate dataset size.

Therefore:

```text
NSA is evidence about temporal mechanism sensitivity,
not proof that Transformers are intrinsically unsuitable for DVS-Lip.
```

## 5.2 SpikGRU2+

Primary paper:

https://openaccess.thecvf.com/content/CVPR2024W/EVW/papers/Dampfhoffer_Neuromorphic_Lip-Reading_with_Signed_Spiking_Gated_Recurrent_Units_CVPRW_2024_paper.pdf

Official code:

https://github.com/manondampfhoffer/SpikGRU-DVSLip

Important facts:

- reported DVS-Lip accuracy: 75.3%;
- model size: approximately 58.6M parameters;
- frontend: Spiking ResNet-18;
- backend: bidirectional SpikGRU2+;
- the recurrent backend uses sigmoid gates and ternary spikes;
- the model is not a causal streaming design because the backend is bidirectional;
- data augmentation and fine-tuning materially affect reported performance.

Therefore SpikGRU2+ proves that **strong gated recurrence can be highly effective**, but it does not isolate gating as the sole source of the gain and it does not demonstrate sub-500k feasibility.

It is an accuracy anchor and architectural prior, not a directly deployable target architecture.

## 5.3 Multiplication-Free Channel-wise PSN

Primary source:

https://papers.nips.cc/paper_files/paper/2025/file/5f40a99efedc1148a061d80201949c84-Paper-Conference.pdf

Important DVS-Lip result:

- Modified Spiking ResNet-18 + Mul-free Channel-wise PSN;
- FC backend with stateful synapses;
- 70.9% DVS-Lip accuracy;
- neuron order \(k=2\) with sawtooth dilations;
- coefficients are structured for multiplication-free / bit-shift computation.

This result is critical because it shows that:

```text
GRU-style gating is not the only path to strong DVS-Lip temporal modeling.
```

A compact, structured temporal neuron can capture rich history with small order and hardware-friendly arithmetic.

The exact model is still ResNet-18 scale; it is not evidence that 70.9% is attainable at 500k parameters.

For this thesis, Mul-free PSN is a central **temporal-core / neuron-design reference**.

## 5.4 Event2Vec

Primary source:

https://arxiv.org/abs/2504.15371

Official code:

https://github.com/Intelligent-Computing-Lab-Panda/event2vec

Current DVS-Lip results reported in the paper:

- 1024 random events: 70.62 ± 1.55%;
- 1024 Batched K-Means++ cluster events: 75.88%;
- reported model parameter storage for DVS-Lip: 18.30 MB.

Do **not** convert 18.30 MB into an exact parameter count unless the parameter precision/storage convention is verified.

Event2Vec also reports:

- a 16-layer DVS-Lip backbone;
- self-supervised pretraining for the best DVS-Lip model;
- significant sensitivity of DVS-Lip performance to the temporal attention choice;
- FoX 75.88% vs GLA 72.35%;
- explicit discussion of long-memory attenuation.

This demonstrates that Transformer/attention-based processing can work very well when representation and temporal dynamics are appropriate.

Therefore:

```text
the evidence does not justify discarding QKTA/attention;
it justifies making temporal memory a first-class component.
```

---

# 6. Architectural thesis hypothesis

The current working architectural hypothesis is:

> **Separate spatial interaction from temporal memory instead of asking QKTA/LIF to do both.**

Conceptual decomposition:

```text
raw DVS events
      ↓
compact temporal representation
      ↓
lightweight spatial extraction / SPEDS
      ↓
QKTA or another efficient spatial mixer
      ↓
compact causal temporal core
      ↓
readout / anytime classification
```

Mathematically:

\[
E_t = \mathcal{E}(\mathcal{X}_{\le t}),
\]

\[
S_t = \mathcal{S}(E_t),
\]

\[
H_t = \mathcal{G}(S_t,H_{t-1}),
\]

\[
\hat y_t = \mathcal{R}(H_{\le t}).
\]

Where:

- \(\mathcal{E}\): event-time representation;
- \(\mathcal{S}\): spatial feature extraction / QKTA spatial interaction;
- \(\mathcal{G}\): explicit causal temporal core;
- \(\mathcal{R}\): classifier/readout.

This decomposition is a working hypothesis, not a claim of novelty by itself.

---

# 7. Role of QKTA and Mini-QKFormer

## 7.1 QKTA is retained as a candidate spatial mixer

Do not remove QKTA merely because its arithmetic cost is small.

A spatial operator that contributes useful feature interaction while costing very little is potentially desirable.

Spike-driven attention literature shows that spike-form attention can often be reduced to masks and sparse additions rather than dense floating-point attention.

The relevant design question is:

> does QKTA contribute enough spatial discrimination per unit cost to justify retaining it?

This is an empirical question.

## 7.2 Mini-QKFormer is not automatically the final model

Mini-QKFormer remains useful because:

- it is already implemented;
- it is compact;
- its behavior is understood;
- the repository has profiling/tests around it;
- previous compression work suggests the spatial transformer path can be made small.

However, the new phase must not assume:

```text
Mini-QKFormer + new encoder = final architecture.
```

Instead:

```text
Mini-QKFormer = baseline spatial architecture
temporal core = explicit design variable
```

## 7.3 Do not mischaracterize temporal mean pooling

The current final temporal mean does **not** make the entire network permutation-invariant.

Current features are history-dependent because LIF state propagates over time:

\[
h_t=f(x_{\le t}).
\]

Therefore:

\[
\frac{1}{T}\sum_t h_t
\]

is not generally invariant to permutation of the original input sequence.

The actual concern is:

> temporal mean can dilute temporally localized discriminative evidence if upstream state does not preserve it strongly enough.

This is why mean vs last-state vs compact recurrent readout is a valid ablation.

---

# 8. Two temporal bottlenecks must remain separate

## 8.1 Representation bottleneck

Temporal discretization can destroy micro-timing before the network sees it.

Fixed binning maps distinct event histories to the same frame tensor.

This is **intra-bin temporal information loss**.

## 8.2 Temporal-core bottleneck

A representation may contain useful timing but the network may lack a sufficiently expressive state to exploit it.

This is **temporal-memory / receptive-field / update-rule limitation**.

The project must never interpret:

```text
richer encoder does not help
```

as proof that the encoding is irrelevant **before verifying that the temporal core can use richer timing**.

That would be a false negative.

## 8.3 Positional distinction for delay mechanisms

A delay bank after destructive coarse binning is mainly a memory mechanism.

A delay bank operating on fine event-time input before compression can be part of the encoder.

Therefore “encoder vs delay” is not a universal dichotomy.

Placement and state semantics matter.

---

# 9. Main scientific questions

The revised project has three ordered questions.

## Q1 — Representation

> What compact causal temporal representation preserves useful DVS-Lip dynamics at acceptable cost?

## Q2 — Temporal exploitation

> What minimal causal temporal state can actually exploit that representation under a sub-million, hardware-aware constraint?

## Q3 — Joint interaction

> Is there a measurable interaction between temporal representation richness and temporal-core capacity?

The strongest constructive result would support:

\[
\boxed{\text{representation determines what survives}}
\]

and

\[
\boxed{\text{temporal dynamics determine what is exploited}}.
\]

This must be tested, not assumed.

---

# 10. Success definition

The project seeks **strong task performance for its scale**, not accuracy indifference.

A model that remains far below comparable compact baselines is not acceptable just because it is efficient.

At the same time, the thesis is not judged solely against 58M-parameter models.

The primary objective is the Pareto front:

\[
(\text{Accuracy},\text{Parameters},\text{SOP},E_H,\text{State},\text{Latency}).
\]

Meaningful results include:

- materially higher accuracy at similar cost;
- similar accuracy with major operation/state reductions;
- improved prefix-AUC / earlier stable recognition;
- good fixed-point behavior;
- strong transfer with minimal temporal recalibration;
- a compact model approaching large-model accuracy far more efficiently.

Not sufficient:

- probe-only gains;
- larger model wins explained by capacity;
- unquantizable temporal state;
- delay distributions that look interesting but do not improve task/cost;
- energy claims ignoring memory;
- DVS-Lip-specific supervision that destroys generality;
- non-causal bidirectional modules in the final streaming candidate.

---

# 11. Parameter-budget policy

The thesis retains an aggressive compact target.

Preferred final target:

```text
~500k trainable parameters
```

But this is now treated as a **target to validate**, not a dogma.

We currently do not have evidence that 500k parameters are sufficient for 70% DVS-Lip accuracy.

Known high-performing public anchors are much larger or have parameter-storage figures well above our target.

Therefore the project must explicitly measure capacity scaling.

## 11.1 Capacity sanity experiment

Once E0 and the training recipe are stable, run a cheap capacity scan:

```text
~0.5M
~1.0M
~2.0M
```

same:

- representation;
- training recipe;
- split;
- temporal-core baseline;
- evaluation protocol.

Initially one seed per size is sufficient.

Interpretation examples:

```text
55 / 57 / 58  → likely architecture/training saturation
55 / 63 / 69  → strong capacity sensitivity
63 / 65 / 66  → 500k regime is promising
```

These numbers are examples of interpretation, not predefined expected results.

## 11.2 Final budget decision

If evidence shows that modestly exceeding 500k produces a major Pareto gain, the scientific owner may revise the final cap.

Any revision must be documented in `DECISIONS.md`.

The project must not silently inflate from 500k to multi-million scale.

---

# 12. Hard design constraints

Unless explicitly revised:

## 12.1 Causality

No future event may affect state/output at time \(t\).

Bidirectional SpikGRU-style recurrence is a literature anchor, not an allowed final streaming mechanism.

## 12.2 Streaming

Final inference must support incremental processing.

Offline training vectorization is allowed only if mathematically equivalent to the causal implementation.

## 12.3 Quantization

No final mechanism may require FP32-only persistent state.

Low-bit/fixed-point feasibility is part of architecture selection.

## 12.4 Hardware awareness

Every temporal mechanism must declare:

- parameters;
- persistent state bits;
- state reads/writes;
- additions;
- multiplications;
- sigmoid/exp/tanh or other nonlinear cost;
- comparisons;
- shifts;
- LUT requirements;
- maximum buffer depth;
- BRAM mapping;
- DSP expectation;
- zero-skip opportunity;
- feedback critical path.

## 12.5 No hidden bidirectionality

Anything using reversed future sequence information is prohibited for the final causal model.

It may be implemented only as an explicitly non-causal upper-bound reference if scientifically justified and approved.

---

# 13. Candidate temporal-core families

The project must avoid a second uncontrolled architecture zoo.

Only a small ordered set is allowed.

## C0 — Current LIF temporal dynamics

Baseline.

This is not “no memory”.

LIF already carries leaky membrane state.

## C1 — Cheap global causal recurrent readout

A compact state after the spatial backbone.

Possible comparison:

```text
temporal mean
last state
diagonal/channelwise gated recurrent readout
```

State complexity:

\[
O(C).
\]

Purpose:

- cheap test of whether explicit gated memory materially improves the task;
- attack temporal evidence dilution without large spatial state.

Do not introduce full GRU matrices by default.

## C2 — Hardware-oriented richer spiking neuron

Candidate families inspired by:

- Mul-free Channel-wise PSN;
- PMSN;
- CE-LIF;
- related compact long-memory spiking dynamics.

The goal is not to reproduce every paper.

The question is whether a richer neuron can improve temporal retention with acceptable arithmetic/state cost.

Mul-free / shift-add mechanisms receive high priority because they align with FPGA constraints.

## C3 — Per-stage / spatially local gated state

Only if C1/C2 indicate that stronger temporal state is beneficial but global state is insufficient.

State may scale as:

\[
O(CHW),
\]

so a hardware card is required **before implementation**.

Prefer channelwise/diagonal gates over dense recurrent matrices.

## C4 — State inside attention / SRWA-like recurrence

Last-resort research direction.

Do not implement until C1–C3 establish a clear need.

It has the highest risk of:

- state traffic;
- Q/K/V coupling;
- difficult attribution;
- complex FPGA datapath;
- duplicated temporal machinery.

---

# 14. Why “gated” is more interesting than “another leak”

Current LIF already computes a leaky recurrence.

Adding another fixed first-order low-pass state:

\[
h_t=\alpha h_{t-1}+x_t
\]

may lengthen memory but does not introduce strongly input-dependent forgetting/update semantics.

A richer minimal recurrence can use:

\[
g_t=\sigma(a\odot x_t+b\odot h_{t-1}+c),
\]

\[
h_t=g_t\odot h_{t-1}+(1-g_t)\odot\phi(x_t).
\]

However, the project must count honestly:

- sigmoid cost;
- elementwise multiplications;
- extra state;
- state memory traffic.

A conventional dense GRU with \(D=H=128\) would add roughly:

\[
3(DH+H^2+H)\approx 98.7k
\]

trainable parameters, already a substantial fraction of a 500k budget.

Therefore the preferred research question is:

> what is the **minimal state-dependent update rule** that recovers most of the temporal advantage without paying full GRU cost?

Possible paths:

- diagonal gates;
- binary/ternary gates;
- shift-add state coefficients;
- sparse update;
- multiplication-free spiking state.

Do not choose one before the architecture sanity experiments.

---

# 15. Literature precedent for spatial-attention + temporal recurrence

## 15.1 RVT

**Recurrent Vision Transformers for Object Detection with Event Cameras**

https://arxiv.org/abs/2212.05598

RVT explicitly separates:

- convolutional prior;
- local/global spatial self-attention;
- recurrent temporal feature aggregation.

This is strong precedent that:

```text
spatial attention + temporal recurrence
```

is a sensible event-vision decomposition.

It also establishes a novelty boundary.

Our novelty cannot simply be “add recurrence to a Transformer”.

## 15.2 TEFormer

**TEFormer: Structured Bidirectional Temporal Enhancement Modeling in Spiking Transformers**

https://arxiv.org/abs/2601.18274

TEFormer explicitly targets temporal fusion in Spiking Transformers and includes:

- forward temporal fusion;
- backward gated recurrent structure.

The backward path is incompatible with the final causal streaming constraint, but the paper further reduces the novelty of a generic “recurrent Spiking Transformer” claim.

Therefore any contribution must be more specific:

```text
causal
compact
hardware-aware
state-minimal
event-stream validated
quantized
transferable
```

---

# 16. Software architecture for the new phase

## 16.1 Explicit representation layer

Create:

```text
src/etsr/encoders/
```

Suggested initial structure:

```text
src/etsr/encoders/
├── __init__.py
├── base.py
├── count.py
├── occupancy.py
├── temporal_pack.py
├── temporal_moments.py
├── time_surface.py
└── profiling.py
```

Do not create speculative files for unimplemented mechanisms.

## 16.2 Explicit temporal-core layer

Create a separate package when the first nontrivial core is implemented:

```text
src/etsr/temporal/
```

Possible minimal structure:

```text
src/etsr/temporal/
├── __init__.py
├── base.py
├── identity_lif.py
├── gated_readout.py
└── profiling.py
```

Add additional core files only when experiments justify them.

Do not bury recurrence inside `MiniQKFormer.forward()`.

## 16.3 Canonical raw event sample

Use a typed contract conceptually similar to:

```python
@dataclass
class EventSample:
    x: Tensor | ndarray
    y: Tensor | ndarray
    t_us: Tensor | ndarray
    polarity: Tensor | ndarray
    target: int
    sample_id: str
    speaker_id: str | int | None
    duration_us: int
    metadata: dict[str, Any]
```

Requirements:

- physical timestamps preserved;
- no hidden duration normalization;
- explicit polarity convention;
- stable sample IDs;
- speaker IDs;
- serializable metadata;
- deterministic loading.

## 16.4 Encoded representation contract

Return metadata in addition to tensor values.

At minimum:

```text
tensor
time_axis
time_unit / timestamps
representation_name
representation_parameters
state_profile
```

Do not pretend every representation is simply a “frame stack”.

## 16.5 Model composition

Preferred logical structure:

```python
encoded = encoder(events)
spatial = spatial_backbone(encoded)
temporal = temporal_core(spatial)
logits = readout(temporal)
```

or a clean `EventClassifier` composition.

Each component must be individually:

- testable;
- profilable;
- replaceable;
- documented.

## 16.6 Stateful API

Stateful components must expose explicit ownership, e.g.:

```python
reset_state(...)
step(...)
forward_sequence(...)
detach_state(...)
```

Required tests:

```text
offline == step-by-step
chunked == step-by-step
reset prevents sample leakage
state shape/precision documented
```

---

# 17. Training recipe is a first-class controlled variable

This is a mandatory addition relative to the previous charter.

A weak training recipe can make every architecture/encoder comparison falsely look uninformative.

Before the main representation experiment, the project must stabilize the baseline training recipe on E0.

## 17.1 Why this matters

SpikGRU2+ reports large effects from spatial/temporal augmentation and fine-tuning.

Event2Vec uses a stronger DVS-Lip recipe including, depending on stage:

- AdamW-style optimization;
- warmup;
- cosine schedule;
- weight decay;
- label smoothing;
- gradient clipping;
- event-coordinate geometric augmentations;
- erasing;
- self-supervised pretraining for its best model.

These recipes cannot be copied blindly because model/input semantics differ, but they prove that training recipe is a major confound.

## 17.2 Baseline recipe stabilization

Use a small, budgeted tuning phase on E0.

Candidates may include:

- optimizer choice;
- learning rate;
- warmup;
- cosine/step schedule;
- weight decay;
- label smoothing;
- spatial event augmentation;
- temporal masking/drop;
- modest event-rate perturbation.

Do not run open-ended hyperparameter optimization.

Predeclare a maximum budget.

Once a reasonable recipe is chosen:

```text
freeze it for controlled representation/core comparisons.
```

## 17.3 Augmentation principles

Training-only augmentation must preserve label semantics.

Validation/test receive no stochastic augmentation.

Every augmentation must record:

- probability;
- parameter range;
- whether it changes physical timing;
- whether it changes event count;
- whether it changes spatial geometry.

Do not mix a new representation with a new augmentation recipe in the same decisive experiment.

---

# 18. Revised phase structure

The project phases are now:

```text
P0  Reproducibility + repository transition
P1  DVS-Lip foundation + literature Pareto + recipe stabilization + capacity sanity
P2  Benchmark validity + architecture sanity + representation×temporal-core interaction
P3  Compact temporal representation screening + final architecture selection
P4  Hardware-aware consolidation + quantization + optional change-driven gating
P5  DailyDVS-200 transfer
P6  Predictive-learning stretch work only if core thesis is already complete
```

The core thesis is P0–P5.

P6 is not required for a strong thesis.

---

# 19. P0 — Reproducibility and transition gate

**Maximum intended duration: 1–2 working days.**

P0 is software trust, not a new research phase.

## Tasks

1. Tag DVS-GC audit complete.
2. Create/reconcile `developer`.
3. Run:
   - `pytest -q`;
   - `ruff check src tests`;
   - `python -m compileall -q src tests`;
   - `make check-scripts`;
   - `git diff --check`.
4. Restore a generic smoke test or remove all stale smoke references.
5. Update README/tree/validation docs.
6. Resolve untracked notebooks.
7. Run one known historical sanity benchmark from the repository.
8. Verify Singularity environment.
9. Verify CUDA and environment logging.
10. Freeze `mechanistic_*` research code.
11. Add this revised charter and child documentation.
12. Ensure no cross-sample LIF state leakage.

## Gate

Proceed only if:

- tests pass;
- documentation matches code;
- smoke training works;
- known baseline behaves plausibly;
- state reset semantics are correct.

No new DVS-GC result is required.

---

# 20. P1 — DVS-Lip foundation

## 20.1 Raw loader

Implement raw-event loading first.

Canonical dataset layer must expose original event timestamps.

Do not make pre-binned frames the dataset truth.

## 20.2 Speaker-disjoint split

Use official test speakers.

Create validation from training speakers.

Preferred development structure:

```text
24 speakers → train
6 speakers  → validation
10 speakers → official test embargo
```

Store exact IDs in a versioned split manifest.

Do not touch official test for architecture/representation selection.

## 20.3 Dataset profile

Generate:

```text
dataset_profile.json
```

with:

- number of samples;
- class counts;
- speaker counts;
- duration distribution;
- events/sample;
- event rate;
- ON/OFF ratio;
- spatial occupancy;
- per-class distributions;
- per-speaker distributions;
- corrupt/missing sample checks.

Use this profile to choose physical temporal scales.

Do not choose \(T\), bin width or delay range based only on convention.

## 20.4 Literature Pareto table

This is mandatory before defining success thresholds.

Required columns:

```text
method
year/venue
ANN/SNN
representation
spatial architecture
temporal architecture
causal?
bidirectional?
streaming?
accuracy
Acc1
Acc2
parameters or parameter storage
timesteps / temporal resolution
training recipe
SOP/FLOP if reported
energy if reported
state if reported
official code
comparability caveats
```

Required anchors include at least:

- MSTP;
- Spiking MSTP;
- SpikGRU2+;
- Mul-free Channel-wise PSN DVS-Lip model;
- Event2Vec;
- TVTA;
- NSA neuron/architecture baselines;
- relevant current event-token models.

Never infer sub-500k feasibility from a paper that does not report that scale.

Never silently convert parameter memory in MB into parameter count unless precision is verified.

## 20.5 Training recipe stabilization

Stabilize E0 before architecture conclusions.

Use a fixed small search budget.

Freeze the chosen recipe afterward.

Store the complete recipe in resolved config and experiment manifest.

## 20.6 Capacity sanity scan

Using E0 and the frozen recipe, evaluate approximately:

```text
0.5M
1.0M
2.0M
```

same architecture family and temporal baseline.

Purpose:

- estimate capacity sensitivity;
- determine whether sub-500k is plausible;
- distinguish architectural saturation from under-capacity.

Initially one seed per size.

This is a diagnostic gate, not a full model-selection sweep.

---

# 21. P2 — Benchmark validity and architecture sanity

P2 exists to ensure that the later encoder comparison is capable of producing interpretable conclusions.

## 21.1 Shortcut validity controls

Before expensive sweeps, run cheap linear/statistical baselines.

### Time-resolved statistics

Examples:

- coarse temporal event counts;
- polarity counts;
- duration/event-rate;
- spatial centroids;
- simple temporal motion descriptors.

### Order-invariant counterpart

Use the same information without temporal ordering.

Purpose:

> verify that DVS-Lip is not trivially solvable by the same shortcut family that invalidated DVS-GC.

Do not turn this into another TDUP.

## 21.2 Minimal temporal dependency control

Compare baseline temporal processing with controlled `NoCrossTime` / NoTD behavior.

Use SDBP only if needed to resolve ambiguity.

Do not use “larger STP gap” as a success objective.

The purpose is only to verify that the current pipeline uses temporal information.

## 21.3 Cheap readout sanity test

Using E0 and frozen training recipe, compare:

```text
temporal mean
last-state readout
compact causal gated readout
```

The gated readout must be small, preferably diagonal/channelwise.

Do not add attention pooling in this first test.

Purpose:

- determine whether the final temporal aggregation is a major bottleneck;
- determine whether explicit state improves DVS-Lip at minimal parameter/state cost.

## 21.4 Important interpretation rule

Temporal mean is not “memoryless”.

The current network already has LIF history.

Therefore the comparison is:

```text
weak implicit LIF temporal core
vs
explicit compact temporal core
```

not:

```text
no memory
vs
memory.
```

---

# 22. Central 2×2 interaction experiment

Before a broad encoder bake-off, run a small experiment that protects against false negatives.

Use:

```text
Temporal representation:
  E_coarse
  E_fine

Temporal core:
  C_LIF
  C_explicit
```

Recommended initial `E_fine`:

- higher-rate count representation, or
- the cheapest fine-time control whose semantics are clear.

Do not start with a highly novel learned encoder.

The four cells:

| | Current LIF temporal core | Compact explicit temporal core |
|---|---:|---:|
| Coarse representation | A | B |
| Fine representation | C | D |

Measure:

\[
\Delta_{\text{interaction}}=(D-B)-(C-A).
\]

Interpretation:

### Positive interaction

Richer timing helps more when explicit temporal memory is available.

This strongly supports joint representation/core co-design.

### Representation main effect only

Fine timing helps even with current LIF.

Proceed to representation optimization.

### Temporal-core main effect only

Memory matters, but richer input timing does not.

Prioritize temporal core rather than exotic encoder.

### Neither helps

Revisit:

- capacity;
- training;
- preprocessing;
- architecture;
- literature anchor reproduction.

Do not conclude “time does not matter”.

This experiment is a central scientific figure candidate.

---

# 23. P3 — Compact temporal representation screening

Only after P2 verifies that the architecture can exploit temporal information should the broader encoder screen drive conclusions.

Candidate family:

| ID | Representation | Main question |
|---|---|---|
| E0 | low-bit count frames | practical baseline |
| E1 | binary occupancy | how important is event multiplicity? |
| E2 | higher-rate count | how important is temporal resolution? |
| E3 | temporal-pack | can fine time survive with fewer serial SNN updates? |
| E4 | fixed continuous-time moments | can compact coefficients preserve intra-window dynamics? |
| E5 | multi-decay time surface | does local decay state give a favorable Pareto trade-off? |

Learned integer-delay encoding is **not** automatically included in broad screening.

It becomes a P3 architectural candidate only if fixed/fine representations demonstrate useful headroom.

## 23.1 Representation families are not mathematically identical

### Fixed moments

Projection/accumulation on known temporal basis functions.

### Time surface

Recurrent decaying local state.

### Delay bank

Explicit access to delayed history.

They may all summarize history, but differ in:

- update law;
- state memory;
- read/write pattern;
- arithmetic;
- causal semantics;
- FPGA datapath.

Do not call them the same mechanism.

---

# 24. Fairness of representation/core comparisons

Changing representation can change input channels and therefore change first-layer capacity.

This is a major confound.

Every row must report:

\[
P,
\quad
N_{\mathrm{MAC}},
\quad
N_{\mathrm{AC}},
\quad
N_{\mathrm{SOP}},
\quad
B_{\mathrm{state}}.
\]

Also:

- input channels;
- feature widths;
- spatial resolution;
- temporal steps;
- GPU hours;
- peak training memory when practical.

## 24.1 Broad screening

Do not make every candidate perfectly iso-parametric before knowing whether it is useful.

Use reasonable native configurations and report real cost.

Eliminate dominated variants.

## 24.2 Finalist comparison

For E0 + top representation(s) + winning temporal core:

1. native comparison;
2. approximately iso-parametric comparison;
3. iso-state or iso-energy comparison where meaningful.

Do not attribute gains to temporal representation if they disappear under reasonable capacity matching.

---

# 25. Screening budget

Avoid a combinatorial explosion.

## Stage A — broad representation screen

One or two seeds.

Validation only.

Primary signals:

- top-1;
- Acc1/Acc2 development proxy if valid;
- prefix-AUC;
- parameters;
- Horowitz energy;
- persistent state;
- GPU hours.

A representation gaining less than roughly 0.5 pp while clearly increasing cost is normally removed from consideration.

This is a screening heuristic, not a significance threshold.

## Stage B — final confirmation

Top candidate(s) + E0:

- three seeds;
- confidence intervals;
- full profiling;
- frozen recipe/core.

Only replicated results drive the final architecture.

---

# 26. Prefix / streaming evaluation

The project cares about evidence accumulation over physical time.

## 26.1 Prefix accuracy

For observed fraction \(\rho\):

\[
A(\rho)=P(\hat y(E_{\le \rho T})=y).
\]

Plot the curve.

## 26.2 Prefix AUC

Primary compact early-recognition metric:

\[
\mathrm{pAUC}=\int_0^1 A(\rho)\,d\rho.
\]

## 26.3 Robust stable decision

If needed:

\[
t_{\mathrm{RSD}}
=
\min_t
\left[
\frac{\sum_{\tau=t}^{T}\mathbf{1}[\hat y_\tau=\hat y_T]}
{T-t+1}
\ge \eta
\right],
\]

with e.g. \(\eta=0.95\).

Treat as secondary until validated.

Do not call diagnostic early recognition an “early exit” unless computation actually stops online and the latency/accuracy trade-off is measured.

---

# 27. P3 architecture decision logic

After P1–P3 evidence:

## Case A — coarse representation already sufficient

Stop encoder innovation.

Focus on temporal core / neuron dynamics.

## Case B — temporal resolution clearly helps

Seek a compact way to preserve it without \(O(T)\) serial BPTT.

Prioritize:

- temporal pack;
- compact fixed state;
- efficient learned delay only if justified.

## Case C — counts beat occupancy

Event multiplicity matters.

Do not force binary-only input.

Use low-bit values if they improve Pareto performance.

## Case D — compact event-time representation wins

Promote it to the main thesis representation.

If fixed state is already strong, do **not** add learning merely for sophistication.

## Case E — explicit temporal core dominates representation choice

Make temporal state the primary contribution.

Representation becomes a supporting design decision.

## Case F — richer representation and richer temporal core interact strongly

This is the preferred scientific outcome.

The architecture becomes a co-designed:

```text
compact temporal representation + lightweight spatial mixer + compact causal temporal state
```

## Case G — all candidates remain far from reasonable Pareto anchors

Do not add more modules.

Investigate:

- capacity;
- training;
- preprocessing;
- crop/resolution;
- baseline architecture;
- implementation correctness.

---

# 28. Learnable delay policy

Learnable delays remain a valid candidate, but they are conditional.

Use them if:

- fine-time information is shown to matter;
- fixed compact representations leave headroom;
- delay memory fits hardware budget.

Prefer discrete/integer delays from the beginning:

\[
d_k\in\mathbb{Z}.
\]

Candidate bank:

```text
{0, 1, 2, 4, 8, ...}
```

Possible design:

```text
overcomplete finite delay bank
→ learned structured selection
→ pruning
→ compact inference bank
```

Required controls:

- delay zero;
- fixed delay bank;
- similarly sized non-delay control.

Do not claim learned temporal placement unless learned delays beat these controls.

Avoid continuous interpolation unless there is strong evidence it is necessary and a credible discretization path exists.

---

# 29. P4 — Hardware-aware consolidation

P4 is mandatory.

A thesis-worthy architecture must survive hardware-oriented consolidation.

## 29.1 Horowitz proxy

Continue to report:

\[
E_H=N_{\mathrm{MAC}}E_{\mathrm{MAC}}+N_{\mathrm{AC}}E_{\mathrm{AC}}.
\]

Document constants and technology assumptions.

Never call it measured FPGA energy.

## 29.2 State and memory metrics

Report:

- persistent state bits;
- state reads;
- state writes;
- buffer depth;
- sparsity;
- estimated BRAM footprint;
- arithmetic precision.

Do not hide recurrent memory traffic.

## 29.3 Hardware card

Every temporal component:

```text
Module:
Purpose:
Trainable parameters:
Persistent state:
State bit width:
Reads/event or query:
Writes/event or query:
Adds:
Multiplies:
Comparisons:
Bit shifts:
Sigmoid/tanh/exp/LUT:
DSP expectation:
BRAM expectation:
Maximum buffer depth:
Feedback path:
Causal:
Natural streaming:
Zero-skip potential:
Quantization status:
Known synthesis risk:
```

The hardware card is required **before** implementing CHW-scale recurrent state.

---

# 30. Quantization policy

Quantization is not a final cosmetic step.

Minimum:

- 8-bit weights;
- 4-bit weights where stable;
- 8-bit temporal state;
- lower state precision where possible;
- fixed-point membrane state.

Use PTQ first where appropriate; use QAT when PTQ degrades the model.

Suggested practical tolerance:

```text
accuracy drop <= 1–1.5 percentage points
```

unless larger loss buys a compelling hardware benefit.

Evaluate quantization effects on:

- final accuracy;
- prefix-AUC;
- firing rate;
- temporal stability;
- gate/state behavior.

---

# 31. Temporal-difference / change-driven gating

This remains a hardware-oriented optional extension after the architecture is stable.

Baseline:

\[
e_t=z_t-z_{t-1},
\]

\[
g_t=\mathbf{1}[\|e_t\|>\theta].
\]

Use names:

```text
temporal-difference gating
change-driven computation
```

not predictive coding.

Possible actions:

- reuse state;
- skip token update;
- skip block;
- suppress low-value computation.

Measure:

\[
\Delta Accuracy,
\quad
\Delta SOP,
\quad
\Delta E_H,
\quad
\Delta State\ Traffic.
\]

A large compute reduction with negligible accuracy loss is a valuable result by itself.

---

# 32. Predictive representation learning — P6 stretch only

Do not make JEPA a required deliverable.

Only attempt prediction after:

- architecture is stable;
- final representation/core are selected;
- quantization is viable;
- hardware profiling is complete;
- DailyDVS transfer is complete or clearly on schedule.

Start simple:

\[
\hat z_{t+\Delta}=g_\phi(z_t),
\]

\[
\mathcal{L}
=
\mathcal{L}_{CE}
+
\lambda\mathcal{L}_{future}.
\]

Mandatory trivial predictors:

- zero predictor;
- last-state predictor;
- simple linear predictor.

If learned prediction does not beat trivial baselines:

```text
stop predictive direction.
```

Only if predictive quality and downstream utility are established may prediction error be used for gating.

A full EMA-target JEPA is future work unless time and results are exceptional.

Maintain terminology:

```text
future prediction != predictive coding
prediction error must play a functional role for a strong predictive-coding claim
```

---

# 33. P5 — DailyDVS-200 transfer protocol

Purpose:

> test mechanism generality without redesigning the architecture.

Freeze:

- representation family;
- temporal-core family;
- number of temporal state components;
- topology;
- bit width;
- compression structure;
- regularization strategy;
- hardware-oriented constraints.

For physical temporal scales \(\tau_k\):

## Rigid transfer

\[
\alpha=1.
\]

## Minimal temporal calibration

\[
\tau_k'=\alpha\tau_k.
\]

Choose a single \(\alpha^\*\) on validation.

Report:

```text
alpha = 1
alpha = alpha*
```

Do not independently retune every delay/time constant.

The value \(\alpha^\*\) is itself an interpretable domain-scale result.

If DailyDVS official splits contain subject overlap, report:

- official/literature-comparable protocol;
- strict subject-disjoint protocol;

clearly separated.

---

# 34. Required literature / software review

Primary sources and official code must be preferred.

## 34.1 DVS-Lip foundation

### MSTP / original DVS-Lip

Tan et al., CVPR 2022
https://openaccess.thecvf.com/content/CVPR2022/papers/Tan_Multi-Grained_Spatio-Temporal_Features_Perceived_Network_for_Event-Based_Lip-Reading_CVPR_2022_paper.pdf
https://github.com/tgc1997/event-based-lip-reading

Use for:

- dataset;
- Acc1/Acc2;
- multi-rate temporal representation;
- original protocol.

### TVTA

https://arxiv.org/abs/2607.08236

Use for:

- temporal modeling before spatial aggregation;
- DVS-Lip novelty boundary;
- viseme/CTC supervision distinction.

### SpikGRU2+

https://openaccess.thecvf.com/content/CVPR2024W/EVW/papers/Dampfhoffer_Neuromorphic_Lip-Reading_with_Signed_Spiking_Gated_Recurrent_Units_CVPRW_2024_paper.pdf
https://github.com/manondampfhoffer/SpikGRU-DVSLip

Use for:

- high-accuracy SNN anchor;
- gated recurrence;
- augmentation;
- sparsity/operation trade-off;
- warning about bidirectionality and model size.

Do not claim it proves small-model GRU superiority.

### Multiplication-Free Channel-wise PSN

https://papers.nips.cc/paper_files/paper/2025/file/5f40a99efedc1148a061d80201949c84-Paper-Conference.pdf

Use for:

- strong DVS-Lip temporal-neuron anchor;
- multiplication-free temporal dynamics;
- low-order history;
- shift-add hardware inspiration.

### Event2Vec

https://arxiv.org/abs/2504.15371
https://github.com/Intelligent-Computing-Lab-Panda/event2vec

Use for:

- event-native representation;
- DVS-Lip 70.62 random / 75.88 clustered anchor;
- parameter-storage efficiency;
- long-memory attention behavior;
- training recipe;
- representation transfer/linear probing.

### Neuromorphic Sequential Arena

https://www.ijcai.org/proceedings/2025/0544.pdf
https://github.com/liyc5929/neuroseqbench

Use for:

- temporal-dependency validation;
- DVS-Lip sensitivity to neuron/core family;
- training-memory-energy trade-offs.

Do not infer exact sub-500k performance from NSA.

## 34.2 Event representation

### Spiking Patches

https://arxiv.org/abs/2510.26614
https://github.com/DTU-PAS/spiking-patches

### FATE

https://arxiv.org/abs/2606.17334

### Event-by-event / SSM references

Use current primary event-stream SSM literature when considering raw-event alternatives.

## 34.3 Spatial/temporal architecture precedent

### QKFormer

https://arxiv.org/abs/2403.16552
https://github.com/zhouchenlin2096/QKFormer

Re-check official reshape/LIF issue before editing SSA/QKTA.

### RVT

https://arxiv.org/abs/2212.05598

Use as key prior art for:

```text
spatial self-attention + temporal recurrent aggregation
```

### TEFormer

https://arxiv.org/abs/2601.18274

Use as current novelty boundary for temporal enhancement in Spiking Transformers.

Its backward recurrent structure is not acceptable for our final causal model.

## 34.4 Delay mechanisms

### BAM-SLDK

DOI:
https://doi.org/10.1088/2634-4386/addb6c

Use for:

- delayed synapses;
- pruning;
- hardware-aware delayed kernels.

### PCL+

https://arxiv.org/abs/2605.12732

Use as novelty boundary for delay + predictive coding.

## 34.5 Hardware / quantization

Read project PDFs and primary sources for:

- Quantized Spike-Driven Transformer;
- Efficient Sparse Hardware Accelerator for Spike-Driven Transformer;
- SimST;
- SpikeVision;
- Spike-Driven Transformer / Spikingformer.

Use them to distinguish:

```text
operation count
memory traffic
actual hardware mapping
```

---

# 35. Required novelty matrix

Before approving the final architecture, maintain:

```text
docs/novelty_matrix.md
```

It must now include at least:

- MSTP;
- TVTA;
- Spiking Patches;
- Event2Vec;
- FATE;
- SpikGRU2+;
- Mul-free Channel-wise PSN;
- NSA PMSN/LTC/CE-LIF references;
- RVT;
- TEFormer;
- BAM-SLDK;
- current Mini-QKFormer;
- proposed representation/core candidates.

Required columns:

```text
input primitive
temporal discretization
spatial architecture
temporal architecture
temporal state
state update
readout
causal?
bidirectional?
asynchronous?
learned temporal parameters?
temporal compression location
spatial compression location
training objective
parameter count/storage
persistent state
main arithmetic
memory behavior
reported task
reported accuracy
streaming?
code
FPGA/neuromorphic relevance
direct overlap
remaining differentiation
```

Do not claim novelty before this matrix is current.

---

# 36. Documentation architecture

Keep the charter stable and operational.

## `docs/PROJECT_CHARTER.md`

This file.

Update only when scientific strategy changes.

## `docs/AGENT_RULES.md`

Strict operating rules.

## `docs/ACTIVE_PLAN.md`

Current phase only:

```text
Current phase
Scientific question
Hypothesis
Tasks
Status
Inputs
Artifacts
Go/no-go
Blocked by
Next decision
```

## `docs/DECISIONS.md`

Append-only.

Each decision:

```text
Date
Decision ID
Question
Options
Evidence
Decision
Why
Rejected alternatives
Reversal condition
Commits/runs
```

## `docs/EXPERIMENT_LEDGER.md`

Important runs only:

```text
Run ID
Commit
Config
Seed
Question
Result
Interpretation
Decision consequence
Artifact
```

## `docs/SOURCES.md`

Each source:

```text
Title
Year/venue
Primary URL
Official code
Status
What we use
What we do NOT claim
Implementation notes
```

## `docs/HARDWARE_NOTES.md`

Hardware cards, bit-width assumptions, state estimates, future FPGA target notes.

## `docs/TRAINING_RECIPE.md`

New required child document.

Track:

- baseline recipe;
- search budget;
- augmentations;
- scheduler;
- optimizer;
- regularization;
- freeze date/decision;
- modifications and reasons.

## `docs/archive/dvsgc/`

Archive/index old DVS-GC narrative docs without deleting provenance.

---

# 37. Agent operating rules

## Rule 1 — inspect before editing

Read code, tests, configs, docs and relevant Git history first.

## Rule 2 — primary source first

Read method/supplement/code before implementing another paper’s mechanism.

## Rule 3 — evidence labels

Use:

```text
FACT
INFERENCE
HYPOTHESIS
DECISION
OPEN QUESTION
```

## Rule 4 — challenge invalid tasks

Stop if an experiment:

- confounds representation and capacity;
- changes training recipe and architecture together;
- violates test embargo;
- breaks causality;
- exceeds state budget;
- duplicates prior art;
- cannot answer a decision.

## Rule 5 — one principal variable per attribution experiment

Do not change representation, temporal core, training recipe and quantization simultaneously.

The planned 2×2 interaction experiment is an explicit exception because the interaction itself is the scientific variable.

## Rule 6 — no test leakage

All design choices use train/validation.

Official test only after freeze.

## Rule 7 — reproducibility is part of the result

Record:

- commit;
- dirty state;
- config;
- seed;
- dataset hash;
- split hash;
- environment;
- parameter count;
- artifact path;
- recipe ID.

## Rule 8 — fail loudly

No silent:

- timestamp clipping;
- duration normalization;
- corrupt-sample dropping;
- split substitution;
- class-order change;
- dependency substitution;
- encoder fallback.

## Rule 9 — hardware honesty

Binary spikes do not imply low energy.

Report:

- state;
- memory;
- arithmetic;
- sparsity;
- precision;
- control.

## Rule 10 — avoid abstraction theater

Build abstractions only when real implementations need them.

## Rule 11 — code clarity

Explain:

- tensor shapes;
- time semantics;
- state lifetime;
- gate semantics;
- surrogate gradients;
- hardware approximations;
- profiling assumptions.

## Rule 12 — preserve negative results

A valid failed architecture is a result.

Update ledger and decision log.

## Rule 13 — do not defend QKTA by attachment

If evidence shows the spatial backbone is the bottleneck, revise it.

Mini-QKFormer is a baseline, not an identity.

## Rule 14 — do not chase GRU by fashion

SpikGRU2+ is a strong anchor but violates our scale/causality constraints.

Extract useful principles, not the full architecture.

## Rule 15 — no “leaky recurrence” without novelty/cost justification

LIF already leaks.

Any additional state must provide a capability or Pareto benefit that ordinary LIF does not.

---

# 38. Git workflow

Integration branch:

```text
developer
```

Push regularly to SMILIES GitLab.

Suggested commits:

```text
chore: close dvsgc audit phase
docs: revise project charter for explicit temporal core
docs: add training recipe and novelty matrix
refactor: introduce raw event sample contract
refactor: separate encoder spatial core and temporal core interfaces
feat: add dvslip raw event loader
feat: add count encoder baseline
feat: add causal gated readout
test: verify temporal core streaming equivalence
profile: add recurrent state traffic metrics
```

Before milestone push:

```bash
pytest -q
ruff check src tests
python -m compileall -q src tests
git diff --check
```

Never commit:

- raw datasets;
- checkpoints;
- credentials;
- large run tensors;
- caches;
- writable Singularity sandboxes.

---

# 39. SMILIES server workflow

Assume Linux over SSH.

Use Singularity for dependencies.

Use `screen` for long experiments.

Before a long run:

1. verify branch/commit;
2. `git status`;
3. run tests;
4. inspect `nvtop`;
5. inspect `htop`;
6. record command/config;
7. launch named `screen`;
8. verify artifact path.

Push code/documentation periodically.

Keep README/source notes current.

---

# 40. Experiment artifact standard

Each important run should support:

```text
run_id/
├── config_resolved.yaml
├── environment.json
├── dataset_manifest.json
├── split_manifest.json
├── recipe.json
├── history.csv
├── metrics.json
├── profile.json
├── hardware_profile.json
├── run.log
└── summary.json
```

Suggested `hardware_profile.json`:

```json
{
  "parameters": {},
  "mac_ops_per_sample": null,
  "ac_ops_per_sample": null,
  "sop_per_sample": null,
  "horowitz_energy_pj": null,
  "persistent_state_bits": null,
  "state_reads_per_sample": null,
  "state_writes_per_sample": null,
  "max_buffer_bits": null,
  "gate_ops_per_sample": null,
  "bit_shifts_per_sample": null,
  "quantization": {},
  "notes": []
}
```

Do not invent memory-energy precision before the target device/memory system is fixed.

---

# 41. Statistical discipline

## Screening

One or two seeds allowed.

Do not over-interpret small deltas.

## Confirmation

Three seeds for final candidates.

Report:

- mean;
- standard deviation;
- confidence intervals where appropriate;
- per-seed values.

## Interaction experiment

For the final 2×2 confirmation, report per-cell:

- mean accuracy;
- pAUC;
- state/cost;
- interaction estimate;
- seed-wise consistency.

Do not declare a mechanistic interaction from a single seed.

---

# 42. Immediate implementation sequence

The implementation agent should proceed in this order.

## Task 1 — repository transition

- tag audit;
- create/reconcile developer;
- add revised charter;
- create/update child docs;
- fix smoke/documentation drift;
- run quality checks.

## Task 2 — novelty/Pareto update

Deep-read:

- MSTP;
- TVTA;
- Spiking Patches;
- Event2Vec;
- SpikGRU2+;
- Mul-free PSN;
- NSA;
- RVT;
- TEFormer;
- FATE;
- BAM-SLDK.

Produce:

```text
docs/novelty_matrix.md
docs/dvslip_pareto.md
```

Do not implement a new temporal module before this is complete.

## Task 3 — raw DVS-Lip loader

Test:

- timestamp monotonicity;
- polarity;
- sample IDs;
- speaker split disjointness;
- deterministic loading.

## Task 4 — baseline encoder API

Implement E0 count representation through the new encoder layer.

Do not alter Mini-QKFormer semantics.

## Task 5 — training recipe stabilization

Use E0 only.

Freeze the recipe after the bounded tuning budget.

## Task 6 — capacity sanity

Run ~0.5M / 1M / 2M pilot.

Document the curve.

## Task 7 — benchmark validity

Shortcut baselines + minimal NoCrossTime.

## Task 8 — readout / temporal-core sanity

Mean vs last state vs compact causal gated readout.

Profile state and arithmetic.

## Task 9 — central 2×2

Coarse/fine representation × LIF/explicit temporal core.

This experiment determines whether broad representation screening is interpretable.

## Task 10 — representation screening

Run E0–E5 with the validated temporal-core configuration.

## Task 11 — final architecture

Evidence decides whether to prioritize:

- fixed compact representation;
- learned integer delay;
- richer spiking neuron;
- compact gated state;
- combination supported by the 2×2.

## Task 12 — P4 consolidation

Quantization, pruning, state/memory profiling, optional change-driven gating.

## Task 13 — DailyDVS transfer

Rigid + single-\(\alpha\) calibration.

## Task 14 — stretch prediction

Only if core thesis is complete.

---

# 43. Escalation rules

Stop and ask the scientific owner when:

- official DVS-Lip protocol is ambiguous;
- primary sources disagree;
- the baseline remains far below literature after recipe stabilization;
- capacity scan indicates 500k may be fundamentally too restrictive;
- temporal core adds excessive state;
- a gated design requires significant dense floating-point arithmetic;
- a representation changes first-layer capacity enough to confound comparison;
- causal streaming equivalence fails;
- official test would need to be accessed early;
- a new paper materially overlaps novelty;
- evidence suggests QKTA itself is the bottleneck;
- evidence suggests the planned temporal representation is irrelevant;
- a proposed architecture cannot plausibly map to FPGA.

Do not solve such conflicts silently.

---

# 44. Autonomous decisions allowed

The AI senior engineer may autonomously:

- improve tests;
- add assertions;
- improve logging;
- refactor duplicated utilities;
- add profiling;
- inspect official repos/issues;
- repair documentation drift;
- propose experiment ordering;
- eliminate clearly dominated screening candidates;
- improve internal APIs;
- recommend dropping an architectural idea if evidence falsifies it.

Requires owner approval:

- dataset change;
- test split change;
- target parameter-budget change;
- main scientific objective change;
- final architecture-family pivot;
- test-embargo change;
- success-criterion change.

---

# 45. Anti-patterns

### Backbone inertia

Do not keep Mini-QKFormer merely because it already exists.

### GRU cargo cult

Do not copy a 58.6M bidirectional SpikGRU because it has high accuracy.

### Metric accumulation

No new probe unless it changes a decision.

### Module shopping

No simultaneous LMU/S4/Mamba/delay/JEPA zoo.

### Accuracy indifference

Efficiency does not excuse a model that is unnecessarily weak.

### Accuracy-only victory

Large capacity does not automatically make a better thesis model.

### Training-recipe confound

Do not compare representations under an unstable baseline recipe.

### Hardware later

Do not postpone state feasibility.

### Binary ideology

Use low-bit counts if they win.

### Temporal-resolution ideology

More bins are not automatically better.

### “Leaky” as novelty

LIF is already leaky.

### Novelty by renaming

FATE-like, Spiking-Patches-like, RVT-like or GRU-like designs must be positioned honestly.

### Predictive-coding inflation

Change detection is not predictive coding.

### Test peeking

Never select on official test.

---

# 46. Thesis contribution hierarchy

## Level 1 — acceptable

A strong compact DVS-Lip SNN baseline with reproducible training and hardware-aware profiling.

## Level 2 — strong

A compact causal temporal core and/or representation that materially shifts the accuracy–energy–state Pareto frontier.

## Level 3 — excellent

A demonstrated representation × temporal-core interaction leading to a compact architecture that approaches larger DVS-Lip systems while remaining quantizable and streaming-compatible.

## Level 4 — very strong

Level 3 plus successful DailyDVS transfer with minimal temporal calibration.

## Level 5 — stretch/publication-oriented

Additional change-driven or predictive computation produces further energy/SOP reduction without meaningful accuracy loss.

A full JEPA/predictive-coding architecture is not required.

---

# 47. Definition of done for the core thesis

The core project is complete when:

- raw DVS-Lip pipeline is reproducible;
- train/val/test speaker protocol is explicit;
- training recipe is stabilized and frozen for comparisons;
- literature Pareto table is current;
- sub-500k capacity assumption has been tested;
- benchmark shortcut/temporal controls are complete;
- Mini-QKFormer/QKTA role has been validated rather than assumed;
- explicit temporal-core sanity experiment is complete;
- central representation × temporal-core interaction has been tested;
- compact representations have been screened fairly;
- final architecture is selected by evidence;
- final DVS-Lip result is replicated across three seeds;
- accuracy, Acc1/Acc2, pAUC, firing, SOP, Horowitz, state and memory metrics are available;
- quantized model is evaluated;
- streaming equivalence is tested;
- FPGA resource implications are quantified;
- DailyDVS transfer is performed;
- documentation and decision provenance are complete.

---

# 48. Decision records to create immediately

## Decision D001 — repository continuation

```text
Continue from refactor/mechanistic-temporal-audit.
Create developer.
Preserve DVS-GC audit history.
Use targeted raw-event/encoder/core refactor rather than rewrite.
```

## Decision D002 — DVS-GC closure

```text
No further DVS-GC research experiments.
DVS-GC remains regression/history only.
```

## Decision D003 — Mini-QKFormer role revision

```text
Mini-QKFormer is no longer assumed to be the final backbone.
QKTA is retained as a candidate efficient spatial mixer.
Temporal memory becomes an explicit first-class architectural variable.
```

Rationale:

- DVS-Lip literature shows large performance sensitivity to temporal dynamics;
- strong results exist with gated recurrence, compact long-memory neurons and event-wise attention;
- no evidence supports a simple Transformer-vs-GRU dichotomy;
- a richer temporal core is required to avoid false-negative representation conclusions.

Reversal condition:

```text
If architecture sanity experiments show current LIF dynamics are already competitive
and explicit temporal state provides no Pareto benefit, retain the simpler core.
```

## Decision D004 — training recipe stabilization

```text
No representation bake-off before E0 training recipe is stabilized under a bounded search.
```

## Decision D005 — parameter target is aggressive, not assumed feasible

```text
~500k remains the preferred final target.
A 0.5M/1M/2M capacity pilot must test whether this scale is viable.
```

---

# 49. Final operating principles

For every architecture idea:

> Does this plausibly improve the accuracy–energy–state Pareto frontier?

For every representation:

> Does it preserve information the temporal core can actually exploit?

For every temporal core:

> Does it add a capability beyond the LIF recurrence already present?

For every gating mechanism:

> What multiplications, nonlinearities, reads and writes does the FPGA actually execute?

For every high literature accuracy:

> What model size, bidirectionality, augmentation, pretraining and causality produced it?

For every experiment:

> What decision changes if the result is positive versus negative?

For every novelty claim:

> What are the nearest primary paper and official implementation?

For every compactness claim:

> Are we reporting parameters only, or also persistent state and memory traffic?

The project is not trying to prove that a fashionable temporal mechanism works.

It is trying to build the **best scientifically defensible compact causal event model** that the evidence and hardware constraints allow.

The active architectural intuition is:

```text
compact event-time representation
        +
efficient spatial interaction
        +
minimal explicit causal temporal state
```

but none of these components is protected from falsification.

Evidence decides.
