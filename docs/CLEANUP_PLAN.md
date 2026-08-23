# Deep-cleanup checklist

Supporto operativo temporaneo per la pulizia approvata il 2026-08-23. Non sostituisce charter,
decision log, protocollo o stato scientifico.

## Vincoli

- [x] Rileggere integralmente `PROJECT_CHARTER.md` e i documenti operativi rilevanti.
- [x] Lasciare invariati charter, regole, decisioni, protocollo e documenti scientifici/storici.
- [x] Preservare l'embargo del test ufficiale DVS-Lip e la riproducibilita dello split.
- [x] Preservare i contratti `EventSample` e `EncodedRepresentation` richiesti dalla roadmap.
- [x] Non anticipare un'interfaccia raw-event generica prima del secondo dataset concreto (D017).
- [x] Mantenere un solo launcher/config-driven entrypoint, riusabile dai dataset futuri.
- [x] Non introdurre framework, registry o diagnostica speculativa.

## Interventi

- [x] Rimuovere dal runtime attivo DVS-GC, smoke sintetico e audit storici gia recuperabili dal tag.
- [x] Ridurre CLI, Makefile, dipendenze e container alla pipeline DVS-Lip attiva.
- [x] Separare l'orchestrazione di training dal dataset senza inventare un contratto futuro.
- [x] Indicizzare DVS-Lip una sola volta e condividere l'indice tra train e validation.
- [x] Eliminare copie e validazioni duplicate nel percorso eventi -> E0.
- [x] Evitare la decodifica dei campioni per costruire il subset di overfit.
- [x] Rimuovere mutazioni diagnostiche dal forward del modello.
- [x] Ridurre metriche e artifact automatici a quelli richiesti dal protocollo corrente.
- [x] Eliminare moduli, test, configurazioni e script rimasti senza chiamanti.
- [x] Consolidare i test sui contratti attivi e aggiungere una regressione sul singolo scan.

## Verifica finale

- [x] `ruff check src tests`.
- [x] `ruff format --check src tests`.
- [x] `python -m compileall -q src tests`.
- [x] `python -m pytest -q`: bloccato esplicitamente; il runtime locale non contiene `pytest`,
  `torch`, `numpy` o `PyYAML`, e Python 3.14 e fuori dal range supportato.
- [x] Sintassi di tutti gli script shell.
- [x] `git diff --check` e ricerca di riferimenti rimasti a file rimossi.
- [x] CLI help e dry-run dei target Make attivi.
- [x] Documenti protetti verificati byte-invariati nel diff.
- [x] Misurare LOC e dipendenze prima/dopo.

## Risultato misurato

- Python di produzione: 6.443 -> 3.044 LOC (-3.399, -52,8%).
- Test: 1.527 -> 965 LOC e 64 -> 36 casi focalizzati sui contratti attivi.
- Diff complessivo incluso questo checklist: 521 righe introdotte, 4.871 rimosse; saldo -4.350.
- Dipendenze runtime dichiarate: 6 -> 3 (`torch`, `numpy`, `PyYAML`); eliminate anche le due
  installazioni legacy fuori da `pyproject.toml`.
- Moduli Python di produzione: 45 -> 30.

## Non obiettivi

- Implementare DailyDVS-200 o indovinarne formato e metadati.
- Aggiungere nuove rappresentazioni, temporal core, metriche o feature di training.
- Riscrivere la documentazione scientifica o cancellare l'archivio DVS-GC.
- Rimuovere il controllo globale D016, il preflight, il profiling del dataset o il manifest di split.
