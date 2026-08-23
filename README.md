# Temporal Event Spiking Research

Repository di ricerca per progettare un modello event-based compatto, causale, quantizzabile e
plausibile per FPGA, con **DVS-Lip** come benchmark primario e **DailyDVS-200** come validazione di
trasferimento.

## Stato attuale

Il progetto è nella fase **DVS-Lip foundation**.

- La fase DVS-Gesture-Chain è chiusa e congelata al tag `dvsgc-audit-complete-2026`.
- Il lavoro attivo avviene sul branch `developer`; il preflight DVS-Lip verificato è al commit
  `489b32f`, mentre `806c0aa` resta lo snapshot congelato della fase precedente.
- La pipeline frame-first DVS-GC, Mini-QKFormer e i temporal/mechanistic audit restano funzionanti ma
  congelati. Il nuovo sviluppo è isolato nel package `etsr.dvslip` e non usa il vecchio factory.
- Il preflight DVS-Lip train-only, il loader raw-event e l'encoder count E0 in tempo fisico sono
  implementati; il temporal core stateful non è ancora implementato.
- L'archivio ufficiale train è stato verificato nel preflight rapido. Poiché la fonte e lo ZIP non
  forniscono mapping speaker, D011 adotta uno split development 80/20 per campione, stratificato e
  riproducibile, dichiaratamente non speaker-disjoint. Split e hash completo del train sono
  verificati; il test resta escluso dallo sviluppo tramite embargo logico, senza spostamento fisico.
- Il P0 hardening ha ripristinato un smoke sintetico limitato, aggiunto regressioni di isolamento LIF
  e introdotto `environment.json`: 35 test e lo smoke CPU end-to-end risultano verificati. Restano
  aperti il sanity storico, SMILIES/CUDA e il remote GitLab.

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
3. [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md);
4. [`docs/DECISIONS.md`](docs/DECISIONS.md), limitatamente alle decisioni rilevanti.

L'indice completo e l'ordine di autorità sono in [`docs/README.md`](docs/README.md).

## Struttura essenziale

```text
configs/              ricetta DVS-Lip, smoke e configurazioni DVS-GC congelate
containers/           ambiente Singularity corrente
docs/                 charter, stato corrente e riferimenti congelati
docs/archive/dvsgc/   documentazione della fase chiusa
notebooks/archive/    notebook storici congelati
scripts/              controlli, launcher SMILIES e helper DVS-GC isolati
src/etsr/dvslip/      implementazione attiva DVS-Lip
src/etsr/{data,...}/  pipeline precedente congelata e utility condivise
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
make check-scripts
git diff --check
```

Il target `make smoke` usa esclusivamente il dataset sintetico e rifiuta configurazioni oltre limiti
stretti prima di avviare training, profiling e audit. È un controllo end-to-end, non un benchmark;
eseguirlo soltanto nel container supportato. Lo stato corrente è in
[`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).

```bash
make smoke
```

Un successo produce `smoke_summary.json` insieme agli artifact di training, profiling e audit. Il
run verificato `smoke_synthetic__20260821_193155__seed7` ha superato il gate; accuracy casuale e
firing profilato nullo confermano che questo resta un test di plumbing, non un benchmark numerico.

La preparazione e il controllo DVS-Lip, che rifiutano un root `test/` e non avviano training, sono:

```bash
make prepare-dvslip-split DVSLIP_TRAIN_ROOT=data/DVS-Lip/train
make preflight-dvslip DVSLIP_TRAIN_ROOT=/absolute/path/to/DVS-Lip/train
make profile-dvslip DVSLIP_TRAIN_ROOT=data/DVS-Lip/train
```

La risoluzione del protocollo senza mapping speaker è registrata nella decisione D011.

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
