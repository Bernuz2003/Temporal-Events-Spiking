# Esecuzione sui server SMILIES

## Regole operative

- lavorare sul branch dedicato alla fase corrente;
- effettuare push periodici;
- usare container Singularity;
- usare `screen` per run lunghi;
- controllare CPU e GPU con `htop` e `nvtop`;
- preferire `/home/users/<username>` per dataset e run intensivi;
- liberare lo spazio locale al termine dell'esperimento.

## Prerequisiti

Richiedere l'abilitazione `fakeroot` e la directory locale:

```text
/home/users/<username>
```

I container scrivibili non devono essere costruiti sul filesystem home di rete.

## Build del container

```bash
cd /home/users/$USER
singularity build --fakeroot \
  temporal-event-spiking.sif \
  /percorso/repository/containers/temporal_event_spiking.def
```

## Avvio interattivo

```bash
singularity shell --nv \
  --bind /percorso/repository:/workspace \
  --bind /home/users/$USER:/local \
  /home/users/$USER/temporal-event-spiking.sif
```

All'interno:

```bash
cd /workspace
python -m pip install -e . --no-deps
pytest -q
```

## Run con screen

```bash
screen -S phase1_order2
```

Dentro la sessione:

```bash
singularity exec --nv \
  --bind /percorso/repository:/workspace \
  --bind /home/users/$USER:/local \
  /home/users/$USER/temporal-event-spiking.sif \
  bash -lc 'cd /workspace && python -m etsr.cli train --config configs/phase1_dvsgc_order2.yaml'
```

Detach: `Ctrl-a`, poi `d`.

Elenco sessioni:

```bash
screen -ls
```

Rientro:

```bash
screen -r phase1_order2
```

## Path di output

È possibile spostare pesi e artifact sul filesystem locale tramite variabili d'ambiente:

```bash
export ETSR_ARTIFACT_ROOT=/local/etsr/artifacts
export ETSR_CHECKPOINT_ROOT=/local/etsr/checkpoints
```

Al termine, copiare gli artifact utili nel repository o nel filesystem persistente e rimuovere i checkpoint locali non necessari.

## Esecuzione della Fase 2

La preparazione del dataset va eseguita una sola volta. I tre training devono condividere la stessa
root e lo stesso split manifest e, su una singola GPU, vanno lanciati sequenzialmente.

Dentro una sessione `screen` e il container:

```bash
cd /workspace
python -m pip install -e . --no-deps
python -m etsr.cli phase2-prepare --config configs/phase2_dvsgc_order2.yaml
python -m etsr.cli phase2-train --config configs/phase2_dvsgc_order2.yaml --seed 42
```

Ripetere il solo comando di training per `123` e `2026`, annotando i tre percorsi `best.pt`. Quindi:

```bash
python -m etsr.cli phase2-audit \
  --config configs/phase2_dvsgc_order2.yaml \
  --checkpoint 42=/path/seed42/best.pt \
  --checkpoint 123=/path/seed123/best.pt \
  --checkpoint 2026=/path/seed2026/best.pt
```

Non cancellare o rigenerare `data/dvsgc_phase2_order2_v1` tra i seed. L’audit verifica gli hash del
dataset e dello split registrati nei checkpoint.
