# Perimetro del Temporal Audit

## Implementato

- dataset factory per DVS-Gesture-Chain e dataset sintetico;
- baseline Mini-QKFormer-like, riscritta in PyTorch e organizzata modularmente;
- split validation separato;
- training con checkpoint selezionato sulla validation;
- test finale unico;
- perturbazioni temporali deterministiche;
- prefix evaluation;
- accuracy, Macro-F1, confusion matrix;
- firing-rate profiling;
- stima MAC, SOP ed energia teorica Horowitz;
- artifact compatti e checkpoint separati;
- container e script per i server SMILIES;
- test automatici senza dataset esterno.

## Non implementato

- encoder di eventi grezzi;
- timestamp intra-bin;
- LMU, S4, Mamba o delayed kernels;
- JEPA o future-event prediction;
- prediction-error gating;
- quantizzazione;
- implementazione FPGA;
- stime di BRAM, buffer o bandwidth;
- sweep estesi di iperparametri.

## Criterio di completamento

Il Temporal Audit è completato quando:

1. la baseline converge su DVS-GC con protocollo riproducibile;
2. il comportamento originale e perturbato è misurato;
3. le metriche temporali permettono di distinguere uso reale dell'ordine da shortcut;
4. firing, SOP ed energia teorica sono disponibili con limiti documentati;
5. i risultati sono replicati sui seed decisivi.
