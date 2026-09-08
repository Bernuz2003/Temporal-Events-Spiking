# Roadmap decisiva

**Aggiornata:** 2026-09-08

## Gate 0 — nessun training lungo

Per F e T sono obbligatori: test di forma e backward, causalità, equivalenza tra elaborazione
sequenziale e step, conteggio parametri/stato/operazioni, e bounded overfit sullo stesso subset.
Un candidato che fallisce viene riparato una volta; non riceve un full run finché il difetto resta.

## Selezione strutturale

1. **Run F e gated-v2, seed 42**, indipendenti e parallelizzabili, ciascuno dopo il proprio overfit.
   F isola il front-end; gated-v2 corregge l'inizializzazione sul backbone baseline, a fixed window.
   Non aggiungere last-event, FIR o augmentation al rilancio gated.
2. Se F raggiunge almeno 46.15 Macro-F1, eseguire **F+T, seed 42**. Anche un F entro −0.5 punti
   dalla baseline, con almeno −25% operazioni affini potenziali e −25% stato, giustifica F+T come
   tentativo di recuperare accuratezza a costo ridotto. Altrimenti eseguire **baseline+T, seed 42**.
   Questo secondo criterio è un compromesso esplorativo, non una superiorità prestazionale.
3. Usare un solo run aggiuntivo per l'ablazione mancante tra F, T e F+T, esclusivamente se può
   cambiare l'attribuzione causale o la candidata.
4. Conservare un solo run di riserva: PLIF per canale oppure temporal packing sul front-end F.
   Il gated corretto fa già parte della prima coppia. La riserva si attiva sulla base del
   collo di bottiglia osservato, non per completare una matrice.

Ogni punto include profiling v4 del best checkpoint sugli stessi 64 campioni validation, con
campionamento per classe e seed fisso del profiler. Prima rigenerare anche i riferimenti storici.
Il costo decide quale modifica replicare quando le metriche sono vicine. Il terzo slot può
recuperare i profili storici; il quarto rimane libero finché manca un'ipotesi indipendente utile.

## Conferma

Quando esiste una candidata:

1. replicare baseline e candidata con due seed nuovi comuni;
2. riportare media, deviazione e valori per seed di Macro-F1/accuracy;
3. confrontare predizioni appaiate, Acc1/Acc2, confusioni e curve di latenza causale;
4. completare il confronto Pareto con parametri, stato persistente, operazioni potenziali, firing
   rate e traffico di stato.

Il nucleo è 3 run di discovery (F, gated-v2, un ramo T), poi 4 run di conferma.
L'eventuale ablazione mancante e la riserva sono condizionali, non impegni a eseguire altri run.
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
