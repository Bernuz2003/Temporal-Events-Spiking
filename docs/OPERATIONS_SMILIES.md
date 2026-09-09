# Operazioni riproducibili su SMILIES

**Aggiornate:** 2026-09-09

Ogni server fisico vede la propria GPU come indice locale `0`. I quattro nomi di sessione screen
sono indipendenti perché vivono su macchine diverse. Prima del lancio, sincronizzare lo stesso
commit pulito su daredevil, mustafar, stmary e kokiu; dati e checkpoint devono restare nei path
configurati.

## Verifica una volta per macchina e commit

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/dataset_workflow.sh dvslip check
```

Il check esegue suite, lint, bytecode, shell syntax e test CUDA/AMP. Non avviare un candidato se
fallisce. Il dataset gate completo non va ripetuto a ogni run.

## Allocazione corrente dei quattro server

### 1. F+TCAP — full candidato principale

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-f-tcap42 -- candidate --config configs/dvslip_f_temporal_capacity.yaml
```

Il workflow esegue bounded overfit, full da pesi nuovi solo se passa, valutazione finale e profilo
v4 del best. Non riusa pesi F o TCAP.

### 2. F+TBR — timing intra-bin compresso

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-f-tbr42 -- candidate --config configs/dvslip_f_tbr.yaml
```

Usa il TBR canonico polarity-agnostic con 8 micro-bin da 6,25 ms in ogni macro-bin da 50 ms. Forma
`[40,1,H,W]`, backbone F e ricetta invariata; non esegue uno sweep di discretizzazione.

### 3. F+Spike-TBR-LIF — filtro dinamico della rappresentazione

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-f-spike-tbr-lif42 -- candidate --config configs/dvslip_f_spike_tbr_lif.yaml
```

Usa `β=0,9`, soglia `1,1`, 8×6,25 ms e reset per macro-finestra. È una ricostruzione paper-aligned,
non una replica di codice ufficiale. Il workflow rifiuta cambi simultanei a F, ricetta,
augmentation o evaluation.

### 4. Diagnostica temporale checkpoint-only B/PLIF

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-temporal-b-plif -- temporal-diagnostic-pair --baseline-config artifacts/dvslip_e0__20260825_211710__seed42/config_resolved.yaml --baseline-checkpoint checkpoints/dvslip_e0__20260825_211710__seed42/best.pt --plif-config artifacts/dvslip_b_plif__20260908_154858_510430__seed42/config_resolved.yaml --plif-checkpoint checkpoints/dvslip_b_plif__20260908_154858_510430__seed42/best.pt --output artifacts/dvslip_temporal_diagnostic_b_plif__20260909
```

Il comando esegue i due checkpoint in sequenza sulla stessa GPU. Non addestra e non modifica i
checkpoint. Crea sottocartelle `baseline/` e `plif/` con:

- `temporal_curve_every_bin.csv`;
- `temporal_curve_event_aligned.csv`;
- `temporal_activity_every_bin.csv`;
- `temporal_activity_event_aligned.csv`;
- `temporal_diagnostic_summary.json`, config e ambiente.

La root contiene `temporal_diagnostic_pair_summary.json` con i delta PLIF−B delle AUC. La
validation completa richiede un forward per checkpoint più riduzioni; la raccolta firing usa hook
aggregati e non conserva tutte le mappe intermedie sulla GPU.

I quattro comandi possono essere assegnati in qualunque ordine ai quattro host. La diagnostica
finirà prima di un full; la GPU liberata resta disponibile per recovery/profiling. Non avviare
`B+T`, E1, una variazione ON/OFF di TBR o MultiGranular-Lite prima della lettura dei tre risultati.

## Monitoraggio e ripresa

```bash
screen -ls
tail -f artifacts/screen/dvslip-f-tcap42.log
screen -r dvslip-f-tcap42
```

`overfit_gate.json` registra il gate; `candidate_workflow.json` collega gate, full e profilo. Se un
full viene interrotto, riprenderlo dal proprio `last.pt` con lo stesso commit pulito:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_training.sh artifacts/<run-id>/config_resolved.yaml dvslip-resume -- --resume checkpoints/<run-id>/last.pt
```

Se fallisce soltanto il profilo:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh profile-recovery -- profile-checkpoint --config artifacts/<run-id>/config_resolved.yaml --checkpoint checkpoints/<run-id>/best.pt --output artifacts/<run-id>/hardware_profile_v4.json --samples 64
```

Una campagna `candidate` rilanciata crea un nuovo overfit e un nuovo full; non è un comando di
resume. Il run gated-v2 manuale resta fermato e non va ripreso.

## Controllo degli output

Un candidato completo deve avere `summary.json`, `history.csv`, predizioni/shortcut, curve prefix,
config/ambiente, `candidate_workflow.json` e `hardware_profile_v4.json`. Confrontare i profili solo
se checkpoint hash, schema v4 e policy di campionamento sono dichiarati. Per i finalisti il profilo
potrà essere esteso a 200 sample; non si deducono joule GPU/FPGA dai contatori Horowitz.
