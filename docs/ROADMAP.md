# Roadmap decisiva

**Aggiornata:** 2026-09-23

## Riferimento e priorità corrente

Il riferimento congelato è **F+DWC-3+TCAP-d8**, E0, tap `[1,2,4,8]`: DVS-Lip seed 42
55,18% Macro-F1, terna 42/43/44 55,54 ± 0,90%; DVS-Gesture seed 42 88,81% Macro-F1.
Sono risultati development, non official-test. La prima discovery è conclusa.

Prima di riprendere augmentation si applica il contratto correttivo di
[`PREDICTIVE_TEMPORAL_PHASE1_AUDIT.md`](PREDICTIVE_TEMPORAL_PHASE1_AUDIT.md). I risultati della
prima esecuzione sono superseded: A1–A4 checkpoint-only devono precedere ogni nuovo training e
selezionano un solo prossimo braccio con il relativo controllo appaiato.

## Ordine vincolante

1. Audit A1–A4 su C0, R0-v2 e S0 archiviati, senza training.
2. Lettura congiunta di autorità dei gradienti, probe di previsione/residuo, movimento delle
   feature e contributo della coda; scelta motivata di un solo braccio.
3. Preflight corretto e controllo appaiato; P-F resta bloccato fino al probe R5 su target e
   orizzonti alternativi.
4. Solo se la firma meccanicistica è coerente, replica del vincitore e controllo ai seed 43/44;
   trasferimento DVS-Gesture del metodo effettivamente supportato dai teacher disponibili.
5. Congelamento di struttura e ricetta; ripresa delle augmentation sulla sola candidata.
6. Eventuale compressione/quantizzazione; official test in un'unica campagna finale.

Non si riaddestra B per questi raffinamenti. Spatial erasing conserva il segnale positivo sul
vecchio riferimento; Temporal Maskout resta negativo nella configurazione provata. Non si
assume additività delle augmentation con un nuovo modello. Run già avviati possono terminare.

## Budget e stop

La riesecuzione non replica l'albero originario. Ogni continuazione ammessa usa 64 epoche e deve
produrre il corredo di evidenza dell'audit; il budget residuo autorizza soltanto il braccio indicato
da A1–A4 e il controllo necessario, prima di un'eventuale conferma multi-seed.

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
