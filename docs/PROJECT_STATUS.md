# Stato corrente

**Aggiornato:** 2026-09-30

**Fase:** selezione finale della memoria condizionale e screen del pretraining predittivo. La prima
esecuzione (sette continuazioni) non è evidenza sulle ipotesi; è archiviata sotto `artifacts/superseded/` e documentata
in [`PREDICTIVE_TEMPORAL_PHASE1_AUDIT.md`](PREDICTIVE_TEMPORAL_PHASE1_AUDIT.md). L'audit
checkpoint-only A1–A4 è completo.

**Prima ondata corretta, seed 42 completata:**

- **D bounded promosso:** 58,31% Macro-F1, +3,14 pp su C0; il vantaggio resta +3,05 pp nella
  finestra tardiva. Il controllo D-static riaddestrato ottiene 55,45%; la diagnostica
  dynamic-vs-constant del checkpoint è stata eseguita.
- **L15 chiuso come risultato di latenza:** +3,54 pp F1 al prefisso mirato di 1,5 s, ma soltanto
  +0,69 pp al punto finale e +1,12 pp di F1-PrefixAUC assoluta rispetto a R0.
- **S0 chiuso come standalone:** il predittore mantiene skill
  valida, ma il +0,79 pp F1 non è risolutivo e l'autorità 0,25 riduce l'accuracy di training finale
  dal 89,10% di C0 al 72,97%.

Gli altri bracci S sono stati ritirati dal repository. S0 resta come controllo predittivo; la
motivazione scientifica sintetica è in [`DECISIONS.md`](DECISIONS.md). Il precedente screen sul
futuro grossolano è stato rimosso mentre viene definito un protocollo corretto.

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

## Cosa ha stabilito l'audit

- La prima S0 non ha mosso la rappresentazione (CKA 0,999 con R0): il suo gradiente ausiliario
  valeva 4–9 × 10⁻⁵ di quello di classificazione.
- L'informazione di classe accessibile linearmente sta nello stage2 (24% contro 5% nello stage1),
  dove parte prevedibile e innovazione ne portano quasi la stessa quantità.
- La coda dopo l'ultimo evento contiene settling discriminativo: +6,71 pp di accuracy fra 1,5 e 2 s,
  +8,42 sulle parole confondibili, per soppressione dei competitori.
- Il vecchio confronto fra target fine e grossolano non definisce da solo uno screen predittivo valido.

## Prossimo passo

Il prossimo braccio strutturale è **Groupwise-D, G=4**, senza comprimere le matrici TCAP dense.
In parallelo è stato definito il
[protocollo dello screen sul futuro sensoriale](FUTURE_SENSORY_SCREEN_PROTOCOL.md), ora
implementato per il fit e la valutazione sull'holdout interno al development-train. Un full di
pretraining seguirà solo un segnale interpretabile rispetto a controlli
preregistrati. Parametri, costo e attività vanno misurati, non usati ora per restringere
artificialmente la capacità del meccanismo. I seed appaiati seguono la selezione del candidato.

Tutte le metriche citate sono development validation. L'implementazione non produce da sola nuova
evidenza empirica; nessun accesso all'official test.
