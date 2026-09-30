# Audit della fase Predictive-Temporal-Coding

**Aggiornato:** 2026-09-30. Questa sintesi sostituisce il protocollo storico della prima
esecuzione. I report grezzi restano negli artifact per tracciabilità; non sono istruzioni per
rilanciare varianti ritirate.

## Validità della prima esecuzione

Le prime continuazioni da 64 epoche non fornivano un confronto attendibile. Il ramp permetteva di
selezionare checkpoint con loss ausiliaria ancora spenta; il preflight controllava i gradienti dei
nuovi moduli senza verificarne l'autorità sul backbone; alcune norme confrontavano insiemi diversi
di parametri. Nella continuazione il backbone era quasi fermo (CKA 0,999 fra rappresentazioni), e
le differenze di F1 erano comparabili alla fluttuazione fra epoche e seed. Gli artifact sono
conservati sotto `artifacts/superseded/`, ma nessuna conclusione architetturale si basa su essi.

## Contratto correttivo mantenuto

| Aspetto | Regola |
|---|---|
| Selezione | Il ramp non conta per selezione né per finestra finale del bounded overfit. |
| Autorità | Misurare rapporto e coseno dei gradienti ausiliari/discriminativi sullo stesso supporto condiviso; ricalibrare a ogni epoca. |
| Confronto | I bracci strutturali partono da zero con ricetta e seed di C0; la continuation L15 ha un controllo di continuation. |
| Diagnostica | Batch stratificati, skill su holdout utterance-disgiunto, varianza del target, causalità e profilo del checkpoint deployabile. |
| Lettura statistica | Riportare best, ultime 16 epoche, curve a prefissi e variabilità fra seed; niente gate di +1 pp su un singolo seed. |

Il report `artifacts/predictive_phase1_audit/phase1_audit.json` contiene le misure A1–A4. Il
preflight automatico blocca un full se inizializzazione, gradienti condivisi o causalità non
soddisfano il contratto. Tutte le metriche qui sotto sono sulla development validation DVS-Lip,
mai sull'official test.

## Esiti validi e implicazioni

| Braccio, seed 42 | Best Macro-F1 | Confronto/interpretazione |
|---|---:|---|
| C0 congelato | 55,18% | Riferimento; altri seed 54,88% e 56,56%. |
| D bounded | 58,31% | +3,14 pp su C0, IC bootstrap appaiato [1,30; 4,99]; ultime 16 epoche +3,05 pp. |
| D-static riaddestrato | 55,45% | Controllo che separa policy dipendente dall'input da gate statici addestrabili. |
| S0 | 55,97% | +0,79 pp su C0, IC [−1,13; 2,70]: non risolutivo. |
| L15 | — | +3,54 pp F1 al prefisso mirato di 1,5 s, ma +0,69 pp al termine rispetto al controllo. |

S0 ottiene skill positiva contro persistenza e media dei ritardi, ma la training accuracy finale
scende a 72,97% contro 89,10% di C0 e il F1 a 1,5 s cala di 5,40 pp. Rendere lo stage2 più
predicibile dal passato può favorire feature lente/settling tardivo senza una decisione migliore o
più precoce. Questo risultato giustifica la conservazione di S0 come controllo scientifico, non
un'ulteriore ricerca locale della stessa loss.

L'audit ha inoltre mostrato che l'informazione di classe è maggiormente accessibile allo stage2
e che la coda senza eventi contiene calcolo discriminativo. Sono diagnosi della rete, non prova
che un dato target predittivo migliori la classificazione. Il precedente screen sul futuro
grossolano è ritirato; il nuovo protocollo va formalizzato prima di implementazione o training.

Le decisioni correnti sono in [DECISIONS.md](DECISIONS.md) e l'ordine sperimentale in
[PREDICTIVE_TEMPORAL_ROADMAP.md](PREDICTIVE_TEMPORAL_ROADMAP.md).
