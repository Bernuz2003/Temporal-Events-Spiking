# Operazioni riproducibili su SMILIES

**Aggiornate:** 2026-09-08

Dalla root del checkout sul server, sincronizzare un commit comprendente codice, config e script.
I launcher richiedono un worktree pulito per registrare una versione riproducibile. I dati e i
checkpoint devono restare nei path configurati; non rigenerare lo split development esistente.
Il container monta `src/`: non serve ricostruirlo per queste modifiche se il precedente funziona.

## Verifica una volta per commit

Con dati/split già validati:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/dataset_workflow.sh dvslip check
```

Esegue suite, test CUDA/AMP a forma DVS-Lip, lint, shell syntax e compilazione. Se container o dati
non sono ancora preparati, usare il workflow `make smilies-build` e
`make smilies-gate DATASET=dvslip`; il gate dati completo include hash e shortcut e non va ripetuto
per ogni candidato. Non avviare le campagne se il check fallisce.

## Run già avviati

F ha superato il bounded overfit ed è entrato nel full training. Gated-v2 ha raggiunto accuracy
1.0 ma ha fallito il vincolo preregistrato `validation_loss < 1.5`; il workflow ha correttamente
evitato il full. Non rilanciarli mentre F è in corso.

`profile-runs` scorre i full run completati, usa le loro config risolte e i loro best, rigenera anche
i due profili v1 e conserva i vecchi file. Include DVS-Gesture. Richiede checkpoint/dataset sul
server; segnala gli assenti in `artifacts/profile_backfill.json` e termina con errore se incompleto,
continuando comunque sugli altri run. Nessun training viene avviato.

## Due nuovi rami indipendenti sui server liberi

Su due macchine fisiche diverse ogni processo usa la GPU locale `0`. Dopo aver sincronizzato lo
stesso commit pulito ed eseguito il check su ciascuna macchina, avviare:

```bash
# Server fisico A
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-b-tcap42 -- candidate --config configs/dvslip_b_temporal_capacity.yaml
```

```bash
# Server fisico B
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-b-plif42 -- candidate --config configs/dvslip_b_plif.yaml
```

Ogni `candidate` esegue il bounded overfit (16×4 campioni, massimo 500 epoche, stop al gate), poi
solo se passa avvia 128 epoche da zero con la ricetta baseline, valutazione finale e profilo v4 del
best su 64 validation. Il gate usa FP32, nessuna augmentation e zero data-loader worker; il full
ripristina esattamente la config originale. Il gate resta quello preregistrato e non viene adattato
al candidato. TCAP e PLIF testano ipotesi diverse e possono procedere in parallelo a F.

## Monitoraggio, ripresa e recupero della sola profilazione

```bash
screen -ls
tail -f artifacts/screen/dvslip-f42.log
screen -r dvslip-f42
```

`overfit_gate.json` contiene l'esito numerico; `candidate_workflow.json` collega gate/full/profilo.
I run ID nei log identificano le directory. Se il full training viene interrotto, usare la sua
config risolta e il suo `last.pt` con lo stesso commit pulito; non usare il last dell'overfit:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_training.sh artifacts/<run-id>/config_resolved.yaml dvslip-resume -- --resume checkpoints/<run-id>/last.pt
```

Dopo una ripresa con `train`, oppure se fallisce solo il profiling, recuperare il profilo senza
ripetere il training:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh profile-recovery -- profile-checkpoint --config artifacts/<run-id>/config_resolved.yaml --checkpoint checkpoints/<run-id>/best.pt --output artifacts/<run-id>/hardware_profile_v4.json --samples 64
```

Il wrapper `train` rimane disponibile per riprese e diagnosi; i nuovi full candidati vanno avviati
tramite `candidate`. Per una ripresa dell'overfit usare la sua config risolta; verificare poi il
gate e non interpretare il suo punteggio come validation indipendente. Una campagna `candidate`
rilanciata da capo crea un nuovo gate e un nuovo full, quindi non è il comando di ripresa.

La tabella finale confronta profili dello stesso schema/campionamento, parametri, MAC/AC potenziali,
attività per layer, membrane/buffer/traffico e proxy aritmetica Horowitz; nessuna misura di joule
GPU/FPGA viene dedotta dai contatori. Per i finalisti estendere il profilo a 200 campioni.
