# Data directory

I dataset e i derivati non sono versionati. Non aggiungere dati grezzi, frame integrati o manifest
generati al repository.

## DVS-Gesture-Chain — frozen legacy

La pipeline esistente usa:

```text
data/dvsgc/download/
data/dvsgc/events_np/train/
data/matched_dvsgc_order2_v1/
```

Le istruzioni complete sono archiviate in
[`docs/archive/dvsgc/datasets.md`](../docs/archive/dvsgc/datasets.md). DVS-GC resta disponibile per
provenienza e regressione, ma non è un target di ricerca attivo.

## DVS-Lip — not implemented

Il percorso locale, il formato sorgente e il manifest non sono ancora definiti nel codice. Non
creare convenzioni implicite. Il futuro loader dovrà preservare gli eventi raw `(x, y, t, polarity)`,
sample ID stabili, speaker ID e timestamp fisici fino all'encoder esplicito. Split e layout saranno
documentati solo dopo verifica del dataset e del protocollo ufficiali.

## Regole

- nessun fallback silenzioso su split o file mancanti;
- nessuna normalizzazione nascosta della durata o dei timestamp;
- manifest di dataset e split versionati negli artifact, non dati grezzi in Git;
- official test DVS-Lip non usato per selezione di ricetta, rappresentazione o architettura.
