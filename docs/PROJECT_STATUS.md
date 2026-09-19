# Stato corrente

**Aggiornato:** 2026-09-19

**Fase:** verifica su SMILIES della ricerca predittiva/condizionale prima della ripresa delle
augmentation. Protocollo, moduli, diagnostiche, workflow ed export deployabile sono implementati
nel branch `Predictive-Temporal-Coding`; nessun nuovo training è stato ancora lanciato.

## Riferimento empirico

La struttura congelata è **F+DWC-3+TCAP-d8**, 501.028 parametri su DVS-Lip, E0 e tap `[1,2,4,8]`.
Il seed 42 raggiunge **55,53% accuracy e 55,18% Macro-F1**. I seed 43/44 raggiungono
54,88/56,56% F1; la media della terna è **55,54 ± 0,90%**, SD campionaria.
Rispetto alla baseline B seed 42 (44,15% F1), il delta appaiato è +11,02 pp.

Il trasferimento DVS-Gesture seed 42 termina a **89,39% accuracy e 88,81% F1**. Non è ancora una
conferma multi-seed Gesture. Metriche e riferimenti agli artifact sono nella
[review predittiva](PREDICTIVE_TEMPORAL_RESEARCH_REVIEW.md).

Le prime augmentation Lip hanno prodotto **50,91% F1 con Temporal Maskout** (−4,27 pp) e
**56,69% con spatial erasing** (+1,52 pp), seed 42. Sono risultati di ricetta diversi dal riferimento
architetturale; non si usano per attribuire un miglioramento alla nuova fase predittiva.
Il [registro augmentation](DATA_AUGMENTATION_RESEARCH.md) conserva analisi e limiti.

## Evidenza temporale riutilizzabile

MG+TCAP resta un candidato più costoso con beneficio limitato sul precedente F+TCAP (+0,80 pp F1)
e beneficio maggiore ai prefissi. Il ramo fine può fornire un target di distillazione, ma la sua
utilità come teacher isolato non è ancora dimostrata.

PLIF resta inferiore a B sul punto finale (−0,47 pp F1), con +3,47 pp di F1-PrefixAUC e circa
+10 pp tra 100 e 300 ms dopo l'ultimo evento. Motiva lo studio delle decisioni precoci;
non autorizza a reinserire PLIF o scegliere un cutoff di coda. Il [ledger](EXPERIMENT_LEDGER.md)
conserva le diagnostiche e i profili storici.

## Prossimo passo

Eseguire il check CUDA/AMP e le diagnostiche P0 della
[`PREDICTIVE_TEMPORAL_ROADMAP.md`](PREDICTIVE_TEMPORAL_ROADMAP.md), quindi il controllo R0.
Solo dopo si aprono i bracci condizionali previsti. Augmentation sospese per nuovi lanci;
risultati e run già avviati conservati.

Tutte le metriche citate sono development validation. L'implementazione non produce da sola nuova
evidenza empirica; nessun accesso all'official test.
