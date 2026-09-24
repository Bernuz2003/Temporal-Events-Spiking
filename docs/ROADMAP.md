# Roadmap decisiva

**Aggiornata:** 2026-09-24

## Riferimento e priorità corrente

Il riferimento congelato è **F+DWC-3+TCAP-d8**, E0, tap `[1,2,4,8]`: DVS-Lip seed 42
55,18% Macro-F1, terna 42/43/44 55,54 ± 0,90%; DVS-Gesture seed 42 88,81% Macro-F1.
Sono risultati development, non official-test. La prima discovery è conclusa.

Prima di riprendere augmentation si applica il contratto correttivo di
[`PREDICTIVE_TEMPORAL_PHASE1_AUDIT.md`](PREDICTIVE_TEMPORAL_PHASE1_AUDIT.md). I risultati della
prima esecuzione sono superseded. Gli esiti scientifici A1–A4 e R5 sono documentati; il bundle di
autorizzazione va rigenerato nello schema 2 prima dei training. La sezione 12 del contratto fissa
bracci, regimi e controlli della riesecuzione.

## Ordine vincolante

1. Rigenerare il bundle A1–A4 nello schema 2: A1 su quattro batch stratificati/64 classi e A2 a
   8192/2048 campioni. Il report schema 1 documenta l'analisi, ma non autorizza training.
2. Seed 42: L15 in continuazione con controllo R0; D, S0 e S1 da zero con la ricetta di C0,
   confrontati con i seed archiviati di C0 (S1 anche con S0). Preflight letto prima di ogni lancio.
3. Seed 43 e 44 per i bracci con firma meccanicistica coerente; controlli di attribuzione
   (gate statici per D, distillazione a finestra piena per L) solo per i bracci positivi.
4. Un'unica fusione strutturata, sorpresa → ampiezza e contenuto → allocazione, solo dopo due
   componenti positivi; poi trasferimento DVS-Gesture del metodo supportato dai teacher disponibili.
5. Congelamento di struttura e ricetta; ripresa delle augmentation sulla sola candidata.
6. Eventuale compressione/quantizzazione; official test in un'unica campagna finale.

Non si riaddestra B per questi raffinamenti. Spatial erasing conserva il segnale positivo sul
vecchio riferimento; Temporal Maskout resta negativo nella configurazione provata. Non si
assume additività delle augmentation con un nuovo modello. Run già avviati possono terminare.

## Budget e stop

La riesecuzione non replica l'albero originario né apre un budget indefinito. Il vecchio tetto,
incompatibile con i bracci scratch da 128 epoche, è sostituito da un programma chiuso: R0/L15/D/S0
al seed 42; S1 solo se S0 mostra la firma predittiva attesa; repliche soltanto dei bracci positivi.
Le continuazioni usano 64 epoche e ogni run produce il corredo di evidenza dell'audit.

Le soglie di screening, attribuzione, fusione e replica sono fissate nel documento operativo.
Non si aprono sweep per salvare un esito negativo. Un bug rende il run non valido; un negativo
valido resta registrato. Grandi teacher, pretraining EMA esteso da zero, nuove rappresentazioni
e nuove famiglie neuronali restano fuori da questa fase.

I risultati devono riportare accuratezza finale, prefissi e profiling del proprio checkpoint;
un miglioramento di latenza o firing non viene presentato come aumento dell'accuracy o riduzione
misurata dell'energia hardware. L'official test resta embargoed.

La [roadmap di validazione e raffinamento](VALIDATION_REFINEMENT_ROADMAP.md) conserva il piano
di trasferimento/augmentation/compressione e la storia della selezione; per l'ordine corrente
prevalgono questo documento e il protocollo predittivo.
