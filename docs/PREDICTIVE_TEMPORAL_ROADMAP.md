# Roadmap Predictive-Temporal-Coding

**Aggiornata:** 2026-09-30. La fase mira a verificare se una memoria condizionata o un
pretraining predittivo aggiungano valore al modello congelato C0. Le decisioni vincolanti sono in
[DECISIONS.md](DECISIONS.md); i difetti della prima esecuzione sono riassunti
nell'[audit](PREDICTIVE_TEMPORAL_PHASE1_AUDIT.md).

## Riferimenti e risultati acquisiti

C0 è F+DWC-3+TCAP-d8, E0, tap `[1,2,4,8]`: DVS-Lip 55,18% Macro-F1 al seed 42 e
55,54 ± 0,90% sui tre seed. Il trasferimento DVS-Gesture seed 42 ottiene 88,81% F1. Sono
risultati development; l'official test resta embargoed.

D bounded ottiene 58,31% F1 al seed 42. Il controllo D-static riaddestrato ottiene 55,45%; il
vantaggio di D richiede ancora conferma fra seed. S0 conserva una skill predittiva valida ma non
un miglioramento robusto del F1, e peggiora il fit discriminativo. L15 migliora il prefisso a
1,5 s, non il risultato finale. Non si combinano automaticamente meccanismi positivi su metriche
diverse.

## Passo 1 — Groupwise-D

Un solo run completo con quattro gruppi. Le matrici TCAP restano dense e la ricetta coincide con
D: l'esperimento isola l'effetto di una policy di ampiezza/allocazione distinta per gruppo. Il
modulo G=1 deve coincidere con D. Prima del full: test, preflight causale e bounded overfit. Dopo:
F1 best e tardivo, prefissi, statistiche dei gate per gruppo, profilo del proprio best. La
separazione delle politiche è necessaria per interpretare il meccanismo, ma da sola non promuove
la variante. Nessuno sweep di G e nessuna compressione prematura.

## Passo 2 — Nuovo screen predittivo

Il disegno è ancora aperto. Prima di implementarlo si fissano esplicitamente il contesto
osservabile, il target, i controlli causali e di capacità, il holdout utterance-disgiunto e la
misura di informazione utile oltre alla sola qualità di ricostruzione. Lo screen precedente non
fornisce un protocollo valido da riutilizzare. Nessun run completo di pretraining parte dalla
sola plausibilità teorica del target.

Uno screen positivo deve mostrare che il segnale previsto è accessibile causalmente e che la
sua utilità supera i controlli rilevanti sullo stesso holdout. Il costo del predittore non è un
vincolo di scoperta: misurarlo, poi comprimere soltanto un meccanismo valido. La BatchNorm che
mescola tempo e batch può far trapelare il futuro in train; il test di causalità va superato sia
in eval sia in train prima di un full.

## Passo 3 — Conferma, trasferimento, raffinamento

Se lo screen supera i controlli, eseguire un solo full con pretraining predittivo e fine-tuning
supervisionato puro, più il controllo a pari budget necessario per attribuire l'eventuale
beneficio al futuro predetto. Una candidata positiva viene confrontata con D e C0 su seed
appaiati; si valutano anche prefissi e profilo. Solo dopo il freeze si riprendono le augmentation
dataset-specific sulla candidata, quindi eventuali compressione e quantizzazione. L'official
test viene usato nella campagna finale.

Non si riaddestra B per ogni raffinamento. Un risultato negativo valido chiude quel ramo; un
bug lo invalida e richiede correzione, non tuning esplorativo.
