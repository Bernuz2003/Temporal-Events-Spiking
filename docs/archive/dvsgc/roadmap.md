# Roadmap

## Temporal Audit

Stabilire se la baseline utilizza davvero l'ordine temporale e costruire un banco di misura affidabile.

## Mechanistic Temporal Representation Audit

Localizzare dove contenuto e ordine diventano disponibili, se vengono conservati e se sono usati
causalmente. Replicare su tre seed, mantenere l’official test sotto embargo e produrre un Temporal
Dynamics Utilization Profile senza modificare l’architettura.

## Intervento architetturale guidato dall’audit

Selezionare una sola modifica coerente con il failure mode osservato: encoder/stato se l’ordine non
emerge, preservazione/gating se si perde, readout se resta disponibile ma non viene usato. Il
confronto dovrà essere quasi iso-parametrico.

## Latent Predictive Training

Pretraining dello stesso frontend tramite predizione latente, seguito da fine-tuning. Valutazione della qualità temporale, non soltanto accuracy.

## Conditional Computation

Simulazione offline del riuso dello stato nelle regioni prevedibili. Implementazione soltanto se emerge un Pareto improvement.

## Hardware awareness

Quantizzazione, integer-only mapping e sintesi FPGA del modello ormai stabilizzato. Le stime hardware dettagliate arrivano solo dopo la validazione del modello.
