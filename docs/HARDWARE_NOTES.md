# Hardware notes

**Status:** accounting contract prepared; no FPGA target or synthesis result yet

## Current verified capabilities

The repository currently reports:

- tagged Conv/Linear MAC or activity-weighted AC operations;
- custom QKTA/SSA mixing AC estimates;
- LIF firing rates;
- Horowitz-style arithmetic proxy using configurable MAC/AC constants.

Default legacy constants are 4.6 pJ/MAC and 0.9 pJ/AC. The output correctly states that it excludes
memory, routing, control and clocking and is not measured FPGA energy.

## Missing accounting required by the new phase

- persistent state bits and precision;
- state reads and writes;
- buffer depth and traffic;
- decay/gate arithmetic;
- comparisons and bit shifts;
- sigmoid/tanh/exp or LUT implementation;
- BRAM/DSP expectation;
- feedback critical path;
- zero-skip feasibility;
- offline, chunked and step execution semantics.

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
