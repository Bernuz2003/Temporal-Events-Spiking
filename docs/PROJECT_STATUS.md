# Stato corrente

**Aggiornato:** 2026-09-24

**Fase:** riesecuzione corretta della ricerca predittiva/condizionale. La prima esecuzione (sette
continuazioni) non è evidenza sulle ipotesi; è archiviata sotto `artifacts/superseded/` e documentata
soltanto in [`PREDICTIVE_TEMPORAL_PHASE1_AUDIT.md`](PREDICTIVE_TEMPORAL_PHASE1_AUDIT.md). L'audit
checkpoint-only A1–A4 e i probe di fattibilità R5 sono completi (sezione 12 dell'audit). Nessun
nuovo training della fase è ancora stato lanciato.

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
- Il target fine non è predicibile dal contesto dello student a nessun orizzonte; il coarse sì.

## Prossimo passo

Quattro bracci, ciascuno da solo sulla propria GPU, seed 42:
**L15** in continuazione da C0 con il controllo **R0**; **D**, **S0** e **S1** da zero con la ricetta
di C0, confrontati con i seed archiviati di C0 (S1 anche con S0). Prima di ogni lancio si legge il
preflight (`predictive-check`), che misura inizializzazione, causalità e autorità dell'obiettivo
ausiliario. I seed 43 e 44 seguono solo per i bracci con firma meccanicistica coerente. Comandi in
[`OPERATIONS_SMILIES.md`](OPERATIONS_SMILIES.md).

Consigliato prima dei lanci: rieseguire l'audit, perché la versione corretta di A1 usa batch
stratificati per classe; la versione registrata misurava una sola parola (audit, sezione 12.1).

Tutte le metriche citate sono development validation. L'implementazione non produce da sola nuova
evidenza empirica; nessun accesso all'official test.
