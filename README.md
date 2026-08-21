# Temporal Event Spiking Research

Repository di ricerca per progettare un modello event-based compatto, causale, quantizzabile e
plausibile per FPGA, con **DVS-Lip** come benchmark primario e **DailyDVS-200** come validazione di
trasferimento.

## Stato attuale

Il progetto è nella fase **P0 — repository transition and reproducibility**.

- La fase DVS-Gesture-Chain è chiusa e congelata al tag `dvsgc-audit-complete-2026`.
- Il lavoro attivo avviene sul branch `developer`, derivato da `806c0aa`.
- Il codice esistente implementa ancora la pipeline frame-first DVS-GC e la baseline diagnostica
  Mini-QKFormer; tali percorsi sono ora marcati frozen.
- Loader raw-event DVS-Lip, encoder espliciti e temporal core stateful non sono ancora implementati.
- Il P0 hardening ha ripristinato un smoke sintetico limitato, aggiunto regressioni di isolamento LIF
  e introdotto `environment.json`; i controlli runtime attendono esecuzione nel container supportato.

La direzione non è “aggiungere DVS-Lip al vecchio dataset factory”. La futura composizione prevista è:

```text
raw events → temporal representation → spatial core → causal temporal core → readout
```

Rappresentazione e memoria temporale sono variabili distinte. Mini-QKFormer/QKTA sono baseline o
candidati spaziali, non architetture finali protette.

## Prima di lavorare

Leggere nell'ordine:

1. [`docs/PROJECT_CHARTER.md`](docs/PROJECT_CHARTER.md);
2. [`docs/AGENT_RULES.md`](docs/AGENT_RULES.md);
3. [`docs/ACTIVE_PLAN.md`](docs/ACTIVE_PLAN.md);
4. [`docs/TASKS.md`](docs/TASKS.md);
5. [`docs/REPOSITORY_STATE.md`](docs/REPOSITORY_STATE.md).

L'indice completo e l'ordine di autorità sono in [`docs/README.md`](docs/README.md).

## Struttura essenziale

```text
configs/              smoke sintetico e configurazioni DVS-GC congelate
containers/           ambiente Singularity corrente
docs/                 charter e registri attivi
docs/archive/dvsgc/   documentazione della fase chiusa
notebooks/archive/    notebook storici congelati
scripts/              smoke e launcher DVS-GC frozen
src/etsr/             implementazione corrente e smoke orchestration
tests/                suite indipendente dai dataset reali
data/                 dati locali non versionati
artifacts/             output di run non versionati
checkpoints/           pesi non versionati e separati dagli artifact
```

## Verifica locale

Ambiente supportato dal package: Python `>=3.10,<3.13` con le dipendenze di `pyproject.toml`.

```bash
make install-dev
make test
make lint
python -m compileall -q src tests
bash -n scripts/*.sh
git diff --check
```

Il target `make smoke` usa esclusivamente il dataset sintetico e rifiuta configurazioni oltre limiti
stretti prima di avviare training, profiling e audit. È un controllo end-to-end, non un benchmark;
eseguirlo soltanto nel container supportato. Lo stato verificato dell'ambiente corrente è in
[`docs/REPOSITORY_STATE.md`](docs/REPOSITORY_STATE.md).

```bash
make smoke
```

Un successo produce `smoke_summary.json` insieme agli artifact di training, profiling e audit.

## Codice storico DVS-GC

I comandi `temporal-audit`, `prepare-matched-dvsgc` e `mechanistic-audit` restano disponibili come
regressione e provenienza. Non autorizzano nuovi esperimenti DVS-GC: la fase può essere riaperta solo
dal responsabile scientifico. Il confine completo è in
[`docs/LEGACY_BOUNDARY.md`](docs/LEGACY_BOUNDARY.md).

## Vincoli non negoziabili

- selezione solo su train/validation; official test DVS-Lip sotto embargo fino al freeze;
- causalità e semantica streaming verificabili;
- confronto controllato, con una variabile principale per esperimento;
- parametri, operazioni, stato persistente e traffico di memoria riportati insieme;
- nessun claim bibliografico o hardware senza fonte/assunzioni esplicite;
- risultati negativi, cambi di strategia e task posticipati registrati, non rimossi.
