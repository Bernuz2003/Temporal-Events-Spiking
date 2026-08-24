# Data directory

I dataset e i derivati non sono versionati. Non aggiungere dati grezzi, frame integrati o manifest
generati al repository.

## DVS-Gesture-Chain — historical data

Vecchi checkout possono aver lasciato dati locali in:

```text
data/dvsgc/download/
data/dvsgc/events_np/train/
data/matched_dvsgc_order2_v1/
```

Le istruzioni complete sono archiviate in
[`docs/archive/dvsgc/datasets.md`](../docs/archive/dvsgc/datasets.md). Il branch attivo non contiene
più il relativo runtime; l'implementazione completa resta al tag `dvsgc-audit-complete-2026`.

## DVS-Lip — verified raw train data

Il percorso verificato è `data/DVS-Lip/DVS-Lip/train`; il codice rifiuta `test/`. Il comando
`make prepare-dvslip-split` crea `data/DVS-Lip/dvslip_development_split.json`. È l'unico manifest
generato necessario: assegna ogni sample ID a train/validation e incorpora seed e algoritmo. Dati e
manifest locale restano ignorati da Git.

## DVS-Gesture — official train preparation

L'archivio verificato e la sua estrazione vivono rispettivamente in
`data/DvsGesture/DvsGesture.tar.gz` e `data/DvsGesture/DvsGesture/`. La preparazione legge soltanto
`trials_to_train.txt` e genera i segmenti usati dal loader sotto `data/DvsGesture/events/train`.
L'archivio e l'estrazione restano entrambi presenti fino al completamento del profilo; poi è
sufficiente conservare archivio e dati preparati.

Il loader preserva gli eventi raw `(x, y, t, polarity)`, sample ID stabili e timestamp fisici fino
all'encoder esplicito. Il mapping speaker non è disponibile: non va inferito e lo split development
deve restare dichiaratamente non speaker-disjoint secondo D011.

## Regole

- nessun fallback silenzioso su split o file mancanti;
- nessuna normalizzazione nascosta della durata o dei timestamp;
- un solo manifest di split auto-descrittivo; nessun manifest duplicato di policy o provenienza;
- official test DVS-Lip non usato per selezione di ricetta, rappresentazione o architettura.
- official test DVS-Gesture non usato durante preparazione o sviluppo.
