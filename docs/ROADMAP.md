# Roadmap decisiva

**Aggiornata:** 2026-09-30

## Riferimento e priorità corrente

Il riferimento congelato è **F+DWC-3+TCAP-d8**, E0, tap `[1,2,4,8]`: DVS-Lip seed 42
55,18% Macro-F1, terna 42/43/44 55,54 ± 0,90%; DVS-Gesture seed 42 88,81% Macro-F1.
Sono risultati development, non official-test. La prima discovery è conclusa.

Il contratto correttivo di [`PREDICTIVE_TEMPORAL_PHASE1_AUDIT.md`](PREDICTIVE_TEMPORAL_PHASE1_AUDIT.md)
ha guidato la riesecuzione; la prima esecuzione resta superseded. Gli esiti della fase successiva
e l'ordine vigente sono in [`DECISIONS.md`](DECISIONS.md), punti 16–19. Le istruzioni operative
precedenti in questa roadmap sono state sostituite dall'ordine seguente.

## Ordine vincolante

1. Groupwise-D con quattro gruppi al seed 42, stessa ricetta di D, gate per gruppo e profilo
   hardware proprio; controllare equivalenza matematica di G=1 con D e superare il preflight e
   il gate di overfit prima del full.
2. Definire e validare un nuovo screen predittivo: contesto, target, controlli e causalità
   train/eval devono essere fissati prima di implementarlo. Nessun run finché il protocollo è aperto.
3. Solo con uno screen interpretabile, un run D con pretraining predittivo e fine-tuning CE.
   Verificare causalità anche in train prima del full. Se positivo, controllo di pretraining sul
   presente a budget di aggiornamenti identico.
4. Freeze del candidato; seed appaiati di candidato, D e C0. Quindi trasferimento,
   augmentation sulla sola candidata e, se necessario, compressione/quantizzazione.
5. Official test in un'unica campagna finale.

Non si riaddestra B per questi raffinamenti. Spatial erasing conserva il segnale positivo sul
vecchio riferimento; Temporal Maskout resta negativo nella configurazione provata. Non si
assume additività delle augmentation con un nuovo modello. Run già avviati possono terminare.

## Budget e stop

La campagna resta chiusa a questi due assi e ai controlli condizionati. La prova di utilità non
impone un tetto prematuro a parametri e compute: una variante positiva viene profilata e solo dopo
eventualmente compressa. Non si aprono sweep per salvare un esito negativo. Un bug rende il run
non valido; un negativo valido resta registrato.

I criteri di lettura e i controlli sono fissati in `DECISIONS.md` prima dei nuovi run.
Grandi teacher, nuove rappresentazioni e nuove famiglie neuronali restano fuori da questa fase.

I risultati devono riportare accuratezza finale, prefissi e profiling del proprio checkpoint;
un miglioramento di latenza o firing non viene presentato come aumento dell'accuracy o riduzione
misurata dell'energia hardware. L'official test resta embargoed.

La [roadmap di validazione e raffinamento](VALIDATION_REFINEMENT_ROADMAP.md) conserva il piano
di trasferimento/augmentation/compressione e la storia della selezione; per l'ordine corrente
prevalgono questo documento e il protocollo predittivo.
