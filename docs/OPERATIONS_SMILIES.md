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

- F+TCAP e F+Spike-TBR-LIF sono già in corso: non rilanciarli.
- F+TBR ha fallito il gate: non avviare manualmente il full.
- La diagnostica B/PLIF è completa in
  `artifacts/dvslip_temporal_diagnostic_b_plif__20260909_v2`.
- Una macchina libera esegue MultiGranular-Lite; la seconda resta riserva.

### F+MultiGranular-Lite — gate e full condizionale

Dopo avere sincronizzato il commit contenente la nuova config ed eseguito il check statico:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-f-mg-lite42 -- candidate --config configs/dvslip_f_multigranular_lite.yaml
```

Il workflow usa E0 coarse a 40×50 ms e un ramo fine ON/OFF a 320×6,25 ms già ridotto a 16×16.
Solo il ramo economico vede 320 step; una convoluzione temporale depthwise causale riduce 8:1 prima
del Transformer. Il full parte da pesi nuovi soltanto se il gate standard passa. Non cambiare
larghezza del ramo, stride spaziale, clock ratio, loss o soglia del gate.

Non avviare `B+T`, E1, una variante TBR o un secondo MultiGranular-Lite sulla macchina rimasta
libera. Il prossimo uso di quella GPU dipende dal risultato finale di F+TCAP.

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
