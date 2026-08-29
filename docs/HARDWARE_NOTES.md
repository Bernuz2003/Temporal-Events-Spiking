# Hardware notes

**Status:** checkpoint profiler implemented; container acceptance pending; no FPGA target or
synthesis result yet

## Current verified capabilities

`profile-checkpoint` is designed to report from a compatible `best.pt` and a deterministic
validation subset:

- analytical Conv/Linear MAC and binary-AC potential;
- binary-AC activity estimates from observed input density;
- dense-potential SOP for QKTA/SSA attention;
- LIF firing rates, FP32 runtime state size, and state accesses;
- observed activation-buffer maxima and inference nonlinearities.

The profiler deliberately emits no Horowitz energy proxy. Numeric format and hardware target are not
fixed, so such a scalar would add apparent precision without representing measured FPGA energy.

## Missing accounting required by the new phase

- selected hardware precision for weights, state and activations;
- scheduled buffer depth and memory traffic;
- bit shifts and exact synthesized decay/reset arithmetic;
- BRAM/DSP expectation;
- feedback critical path;
- zero-skip feasibility;
- offline, chunked and step execution semantics.

Current state bits, reads/writes, decay/reset operations and activation-buffer maxima describe the
executed FP32 model. They are profiling evidence, not a synthesized memory schedule.

No stateful temporal candidate may be called hardware-efficient from parameter count or spike rate
alone.

## Hardware card template

```text
Module:
Purpose:
Implementation commit:
Input/output/event rate:
Trainable parameters:
Weight precision:
Persistent state elements:
State bit width:
Persistent state bits:
Reads per event/query/sample:
Writes per event/query/sample:
Adds:
Multiplies:
Comparisons:
Bit shifts:
Sigmoid/tanh/exp/LUT:
Maximum buffer depth/bits:
DSP expectation:
BRAM expectation:
Feedback path / critical-path risk:
Causal:
Natural streaming:
Offline-step equivalence:
Zero-skip potential and assumption:
Quantization status and accuracy delta:
Known synthesis risk:
Measurement vs estimate boundary:
```

The card is mandatory before implementing state that scales as `O(C*H*W)`.

## Quantization gate

Minimum evaluation for a finalist:

- 8-bit weights, then 4-bit where stable;
- 8-bit temporal/membrane state, then lower where plausible;
- PTQ first when appropriate, QAT if needed;
- final accuracy, prefix-AUC, firing and temporal stability reported together;
- accuracy loss around 1–1.5 percentage points is a guideline, not a hidden pass criterion.

## Open decisions

- **OPEN QUESTION:** target FPGA family, clock and memory hierarchy.
- **OPEN QUESTION:** numeric format for accumulator and temporal state.
- **OPEN QUESTION:** whether zeros can actually suppress memory/arithmetic in the selected datapath.

Memory-energy precision must remain unset until these questions are resolved.
