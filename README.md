# Temporal Event Spiking Research

Repository di ricerca per studiare **quanto e come le dinamiche temporali dei dati DVS vengano perse o sottoutilizzate** nelle pipeline Spiking Transformer.

La baseline iniziale è una Mini-QKFormer pulita e modulare, usata esclusivamente come **strumento di confronto**. Il repository non assume che l'architettura finale debba rimanere QKFormer.

## Obiettivo della release `phase-1`

Questa prima release implementa soltanto il necessario per il **Temporal Audit**:

1. addestrare una baseline su DVS-Gesture-Chain con split train/validation/test separati;
2. verificare se il modello utilizza realmente l'ordine temporale;
3. misurare accuratezza, Macro-F1, curve di riconoscimento sui prefissi, firing rate, SOP stimate ed energia teorica secondo Horowitz;
4. confrontare il comportamento su sequenze originali e perturbate temporalmente;
5. produrre artifact compatti, riproducibili e facilmente confrontabili.

Non sono implementati in questa release nuovi encoder asincroni, LMU, delayed synapses, JEPA o predictive coding. Questi elementi rimangono **direzioni di ricerca**, da introdurre soltanto quando gli esperimenti precedenti ne giustificheranno il costo.

## Struttura essenziale

```text
configs/           configurazioni riproducibili
containers/        container Singularity per i server SMILIES
docs/              visione, dataset, metriche e protocollo
scripts/           comandi di lancio essenziali
src/etsr/          codice Python modulare
tests/             test veloci e indipendenti dai dataset reali
artifacts/         metriche, JSON, CSV e figure (ignorati da Git)
checkpoints/       pesi dei modelli, fisicamente separati (ignorati da Git)
data/              dataset locali (ignorati da Git)
```

## Installazione locale rapida

Python 3.10 o 3.11 è raccomandato.

```bash
make install-dev
pytest -q
```

`make install-dev` installa `dvsgc==0.1.2` separatamente con `--no-deps`, perché i suoi metadata PyPI vincolano una vecchia versione di SpikingJelly non adatta all'ambiente Python corrente. Il progetto usa `spikingjelly==0.0.0.0.14`; il container applica la stessa procedura. Per DVS-Gesture-Chain servono inoltre i file originali DVS128 Gesture. `bash scripts/prepare_dvsgc.sh` prepara la directory e verifica i quattro file richiesti dal dataset.

## Smoke test senza dataset

```bash
python -m etsr.cli train --config configs/smoke.yaml
```

Il comando addestra per una sola epoca su un dataset sintetico e verifica l'intera pipeline: training, validation, checkpoint, test, profiling e artifact.

## Addestramento Phase 1

Configurazione iniziale controllata, catene di due gesti e tre primitive senza ripetizione:

```bash
python -m etsr.cli train --config configs/phase1_dvsgc_order2.yaml
```

Configurazione più complessa, da usare dopo la validazione del protocollo:

```bash
python -m etsr.cli train --config configs/phase1_dvsgc_chain4.yaml
```

## Temporal audit

```bash
python -m etsr.cli audit \
  --config configs/phase1_dvsgc_order2.yaml \
  --checkpoint checkpoints/<RUN_ID>/best.pt
```

L'audit valuta:

- sequenza originale;
- inversione completa del tempo;
- permutazione deterministica dei timestep;
- inversione dell'ordine di segmenti temporali, con target mantenuto;
- inversione dei segmenti con rimappatura della classe, quando disponibile;
- accuratezza osservando prefissi crescenti della sequenza.

L'inversione a segmenti è una **sonda approssimata**: il dataset pubblico non espone i confini esatti dei gesti concatenati. La limitazione è documentata negli artifact.

## Artifact e checkpoint

Per ogni run:

```text
artifacts/<RUN_ID>/
├── config_resolved.yaml
├── history.csv
├── run.log
├── summary.json
├── test_metrics.json
├── confusion_matrix.png
├── profile.json
└── audit/                    creato dal comando audit

checkpoints/<RUN_ID>/
├── best.pt
└── last.pt
```

I pesi non vengono mai salvati nella directory degli artifact.

## Server SMILIES

Consultare [`docs/smilies_setup.md`](docs/smilies_setup.md). In sintesi:

```bash
singularity build --fakeroot temporal-event-spiking.sif containers/temporal_event_spiking.def
screen -S phase1
singularity exec --nv \
  --bind "$PWD:/workspace" \
  --bind /home/users/$USER:/local \
  temporal-event-spiking.sif \
  bash -lc 'cd /workspace && python -m etsr.cli train --config configs/phase1_dvsgc_order2.yaml'
```

Per staccarsi da `screen`: `Ctrl-a`, poi `d`.

## Principi del repository

- una domanda scientifica per esperimento;
- test set usato una sola volta sul miglior checkpoint di validation;
- primo run di sviluppo con un seed, repliche multi-seed solo per risultati decisivi;
- metriche salvate in formati semplici e leggibili;
- nessuna dipendenza da TensorBoard o servizi esterni;
- componenti sostituibili tramite factory e configurazione YAML;
- branch operativo previsto: `developer`.

La visione completa è descritta in [`docs/vision.md`](docs/vision.md); l’alberatura commentata è in [`docs/repository_tree.md`](docs/repository_tree.md) e i controlli eseguiti sono riportati in [`docs/validation.md`](docs/validation.md).
