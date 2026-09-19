# Roadmap decisiva

**Aggiornata:** 2026-09-19

## Riferimento e priorità corrente

Il riferimento congelato è **F+DWC-3+TCAP-d8**, E0, tap `[1,2,4,8]`: DVS-Lip seed 42
55,18% Macro-F1, terna 42/43/44 55,54 ± 0,90%; DVS-Gesture seed 42 88,81% Macro-F1.
Sono risultati development, non official-test. La prima discovery è conclusa.

Prima di riprendere augmentation si apre soltanto la fase delimitata da
[`PREDICTIVE_TEMPORAL_ROADMAP.md`](PREDICTIVE_TEMPORAL_ROADMAP.md). La
[review](PREDICTIVE_TEMPORAL_RESEARCH_REVIEW.md) ne espone motivazioni e limiti.

## Ordine vincolante

1. Diagnostiche causali e di predicibilità sui checkpoint esistenti; verifica del costo.
2. Controllo di continuazione R0 dal checkpoint congelato, stessa recipe dei candidati.
3. Supervisione futura cross-resolution e TCAP dinamico, isolati; controlli attributivi solo
   dove l'esito giustifica proseguire. Fallback di supervisione dei prefissi motivato da PLIF.
4. Errore predittivo per il routing solo se la diagnostica ne sostiene l'utilità, con controllo
   che riceve la stessa auxiliary loss. Nessuna regola imposta «alta sorpresa → meno memoria».
5. Unica fusione dei componenti positivi; replica del solo vincitore e R0 ai seed 43/44;
   trasferimento DVS-Gesture del metodo effettivamente supportato dai teacher disponibili.
6. Congelamento di struttura e ricetta; ripresa delle augmentation sulla sola candidata.
7. Eventuale compressione/quantizzazione; official test in un'unica campagna finale.

Non si riaddestra B per questi raffinamenti. Spatial erasing conserva il segnale positivo sul
vecchio riferimento; Temporal Maskout resta negativo nella configurazione provata. Non si
assume additività delle augmentation con un nuovo modello. Run già avviati possono terminare.

## Budget e stop

Il protocollo dettagliato ammette al massimo **8 continuazioni da 32 epoche per lo screening**,
più 4 per la conferma appaiata Lip e fino a 2 per il trasferimento Gesture. I rami sono
condizionali, non una lista da eseguire integralmente. Il tetto di GPU-ore della discovery
include probe, gate e teacher: tre volte il training del C0 seed 42, circa 26,92 ore dal suo log.
Throughput e hardware del server vanno registrati; numero di epoche e tempo effettivo non coincidono.

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
