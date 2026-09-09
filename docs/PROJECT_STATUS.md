# Stato corrente

**Aggiornato:** 2026-09-09

**Fase:** selezione strutturale avanzata; F+TCAP in corso, TBR canonico chiuso al gate,
MultiGranular configurato per prova di capacità e controllo Pareto

## Verità sperimentale corrente

La baseline B raggiunge 44,81% accuracy e 44,15% Macro-F1. F raggiunge 46,88/46,15 con 431.076
parametri e un profilo nettamente più leggero. B+TCAP è il miglior 500k-class corrente con
48,38/48,12 e migliora B di 3,57/3,97 punti. B+PLIF termina a 43,74/43,68, quindi non è candidato
per il punteggio finale. I controlli 1M e 2M raggiungono F1 49,38 e 51,81 ma servono come upper
bound di capacità. Tabelle complete di prestazione, PrefixAUC e hardware sono in
`EXPERIMENT_LEDGER.md`.

Il profilo F conferma un trade-off favorevole: rispetto a B riduce SOP potenziali del 47,64%, MAC
multivalore del 18,49%, stato e traffico del 56,61%, Horowitz attività del 15,41% e densa del
30,60%. Il firing sale da 0,0585 a 0,1352, quindi il risparmio viene dalla geometria del front-end,
non dalla sparsità. TCAP abbassa firing e AC attività ma aggiunge 251,66 M MAC multivalore e porta
la Horowitz attività a +14,66% rispetto a B.

## Fenomeno temporale PLIF

PLIF supera B di 6,66 punti F1 a 1 s e 8,32 a 1,5 s, poi termina 0,47 punti sotto. Non perde
prestazione nella coda: cresce di 4,93 punti da 1,5 a 2 s, mentre B cresce di 13,72. Poiché il 99,4%
dei sample validation è già terminato entro 1,5 s, il fenomeno riguarda soprattutto la dinamica
post-evento e la normalizzazione del mean. I τ appresi sono eterogenei, con τ medio 6,227 nello
spike-attention finale e τ medi 1,442/1,672 nelle trasformazioni q/proj.

La diagnostica è completata. PLIF migliora F1 PrefixAUC di 3,47 pp e mantiene circa +10 pp F1 tra
100 e 300 ms dopo l'ultimo evento. Le due normalizzazioni del mean sono quasi equivalenti: il
fenomeno dipende dagli stati profondi persistenti. Nessun cutoff PLIF supera però il F1 finale di B,
quindi l'allineamento event-aware resta spiegazione oracle e non diventa una policy di readout.

## Implementazione pronta

- `configs/dvslip_f_temporal_capacity.yaml`: front-end F più mixer TCAP a ritardi 1/2/4;
- `configs/dvslip_f_tbr.yaml`: F con TBR canonico, 8 micro-bin da 6,25 ms per ciascun macro-bin
  da 50 ms;
- `configs/dvslip_f_spike_tbr_lif.yaml`: stesso contratto con filtro LIF paper-aligned,
  `β=0,9` e soglia `1,1`;
- `configs/dvslip_f_multigranular_capacity.yaml`: E0 coarse più ramo 320×32×32, downsampling
  appreso, mixing temporale MIMO 8:1 e fusione residua appresa;
- `configs/dvslip_f_multigranular_lite.yaml`: lo stesso encoder e ramo parametrico a 320×16×16,
  mixing depthwise e fusione additiva;
- comando `temporal-diagnostic`: quattro CSV, summary, config e ambiente, nessun training;
- workflow `candidate`: blocca TBR ai valori DVS-Lip pubblicati e conserva F, ricetta,
  augmentation, split ed evaluation.

TBR e Spike-TBR conservano `T=40` e producono un canale perché la formulazione pubblicata scarta
la polarità. TBR conserva l'occupazione binaria a 6,25 ms, ma perde la molteplicità nello stesso
micro-bin. La ricostruzione Spike-TBR dichiara due scelte non verificabili contro codice ufficiale:
polarità ignorata e reset della membrana a ogni finestra da 50 ms, coerente con l'inizializzazione
per `ΔT` dell'algoritmo pubblicato.

## Prossima acquisizione di evidenza

F+TCAP e F+Spike-TBR-LIF sono in corso. All'epoca 48 F+TCAP raggiunge 39,72% F1 validation ed è
nettamente promettente; all'epoca 38 Spike-TBR è a 8,76% e soffre un input sovrafiltrato. F+TBR ha
fallito il gate pur raggiungendo accuracy quasi perfetta, perché la loss validation non è scesa
sotto 1,5. Non si aggira il gate e non si aprono variazioni TBR.

Le due configurazioni MultiGranular possono ora attraversare il normale workflow `candidate`.
Capacity è la prova falsificabile dell'ipotesi; Lite è il controllo di compressione. Entrambe sono
causali e lasciano il Transformer principale a 40 step. B+T resta rinviato alla compressione
post-freeze e B+E1 resta sospeso.

L'official test resta inutilizzato. Tutti i nuovi full sono seed 42 e servono alla selezione; la
robustezza richiederà due nuovi seed comuni per B e candidata finale prima della fase di
augmentation.
