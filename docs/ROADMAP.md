# Roadmap decisiva

**Aggiornata:** 2026-09-08

## Gate 0 — nessun training lungo

Per ogni candidato sono obbligatori: test di forma e backward, causalità, equivalenza tra
elaborazione sequenziale e step, conteggio parametri/stato/operazioni, e bounded overfit sullo
stesso subset.
Un candidato che fallisce viene riparato una volta; non riceve un full run finché il difetto resta.

## Selezione strutturale — prove indipendenti

1. **F e gated-v2, seed 42**, indipendenti e parallelizzabili, ciascuno dopo il proprio overfit.
   F isola il front-end; gated-v2 corregge l'inizializzazione sul backbone baseline, a fixed window.
   Non aggiungere last-event, FIR o augmentation al rilancio gated.
2. **B+TCAP, seed 42:** MIMO FIR con ritardi 1/2/4 nei due punti a bassa risoluzione. È la prova di
   capacità dell'interazione temporale esplicita, non la candidata hardware finale.
3. **B+PLIF, seed 42:** τ apprendibile per feature channel/head in tutti i LIF. Isola l'adattività
   della memoria neuronale senza aggiungere buffer o trasformazioni del backbone.
4. I quattro rami rispondono a ipotesi distinte. Possono occupare quattro server fisici senza
   attendere F; il parallelismo non autorizza altre combinazioni o sweep.

Ogni punto include profiling v4 del best checkpoint sugli stessi 64 campioni validation, con
campionamento per classe e seed fisso del profiler. Prima rigenerare anche i riferimenti storici.
Il costo decide quale modifica replicare quando le metriche sono vicine.

## Decisione dopo i quattro rami

- Se TCAP migliora di almeno +2 punti Macro-F1, comprimerlo: prima FIR depthwise T; un solo livello
  intermedio cross-channel è ammesso soltanto se T perde il segnale.
- Se PLIF migliora, trasferirlo su F soltanto se F resta Pareto-competitiva.
- Se entrambi migliorano, confrontarli separatamente e combinarli soltanto se una replica o
  un'analisi per layer indica complementarità; il primo seed non basta.
- Se TCAP fallisce, non eseguire la scala di compressione. Se PLIF fallisce, non provare altri τ,
  subset di layer o inizializzazioni durante discovery.
- Se F raggiunge almeno 46.15 Macro-F1, trasferire il miglior meccanismo temporale su F. Anche un F
  entro −0.5 punti dalla baseline, con almeno −25% operazioni affini e stato, può ricevere un solo
  tentativo di recupero.

## Conferma

Quando esiste una candidata:

1. replicare baseline e candidata con due seed nuovi comuni;
2. riportare media, deviazione e valori per seed di Macro-F1/accuracy;
3. confrontare predizioni appaiate, Acc1/Acc2, confusioni e curve di latenza causale;
4. completare il confronto Pareto con parametri, stato persistente, operazioni potenziali, firing
   rate e traffico di stato.

Il nucleo aggiornato è 4 run indipendenti di discovery: F, gated-v2, B+TCAP e B+PLIF. La successiva
compressione/trasferimento usa al massimo due run condizionali. Seguono le repliche comuni soltanto
dei finalisti; l'eventuale ablazione mancante non è un impegno automatico.
Non combinare automaticamente F+T+gated: richiede evidenza che distingua il contributo dei moduli.

## Dopo il freeze

Augmentation e convergenza si provano sulla candidata congelata con una baseline di controllo.
La prima coppia ammessa è la ricetta già implementata di temporal masking più spatial erasing. Si
prosegue solo se migliora la validation senza degradare la curva di latenza. Quantizzazione e
teacher leggeri vengono dopo; pretraining, JEPA, predictive coding e grandi teacher restano fuori
dal budget principale.

## Stop rule

Si congela la migliore soluzione confermata quando un nuovo run non può più distinguere tra le due
ipotesi finaliste o quando il guadagno atteso non giustifica una replica multi-seed. Non si spendono
run per scegliere valori fini di clipping o learning rate prima del freeze.
