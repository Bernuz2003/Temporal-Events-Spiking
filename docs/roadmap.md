# Roadmap

## Fase 1 — Temporal Audit

Stabilire se la baseline utilizza davvero l'ordine temporale e costruire un banco di misura affidabile.

## Fase 2 — Representation Audit

Confrontare il frame count standard con una sola sonda time-aware a costo controllato. Obiettivo: capire se il binning elimina informazione utile.

## Fase 3 — Temporal State

Selezionare un solo stato temporale candidato, più ricco del LIF ma compatto. Confronto quasi iso-parametrico con la baseline.

## Fase 4 — Latent Predictive Training

Pretraining dello stesso frontend tramite predizione latente, seguito da fine-tuning. Valutazione della qualità temporale, non soltanto accuracy.

## Fase 5 — Conditional Computation

Simulazione offline del riuso dello stato nelle regioni prevedibili. Implementazione soltanto se emerge un Pareto improvement.

## Fase 6 — Hardware awareness

Quantizzazione, integer-only mapping e sintesi FPGA del modello ormai stabilizzato. Le stime hardware dettagliate non appartengono alle prime fasi.
