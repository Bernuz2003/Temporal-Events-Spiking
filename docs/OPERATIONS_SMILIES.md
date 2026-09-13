# Operazioni riproducibili su SMILIES

**Aggiornate:** 2026-09-13

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

## Diagnostica checkpoint-only dei tap TCAP

Il comando valuta sequenzialmente checkpoint intatto, tap 1/2/4 azzerati singolarmente e tutta la
storia azzerata. Non addestra e non modifica il checkpoint su disco.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-tcap-taps42 -- tcap-tap-diagnostic --config artifacts/dvslip_f_temporal_capacity__20260909_124154_088394__seed42/config_resolved.yaml --checkpoint checkpoints/dvslip_f_temporal_capacity__20260909_124154_088394__seed42/best.pt --output artifacts/dvslip_tcap_tap_diagnostic__20260913
```

L'output centrale è `tcap_ablation_summary.json`; i CSV separano curva ogni 50 ms, risultati finali,
predizioni appaiate e norme delle matrici di ritardo. `intact_minus_ablated.macro_f1` è il costo F1
della rimozione: solo se `without_delay_4` vale almeno 0,01 e non è inferiore al tap 2 si autorizza
il futuro run con ritardo 8.

La diagnostica del 13 settembre soddisfa il criterio: rimuovere `d=4` costa 51,60 pp F1 contro
49,79 pp per `d=2`. Il risultato autorizza un solo candidato `[1,2,4,8]`; resta una perturbazione
checkpoint-only e non stima direttamente il guadagno del tap nuovo.

## Ultima ondata prima del freeze

Su tre server distinti si eseguono in parallelo le repliche F+TCAP seed 43/44 e il trasferimento
strutturale F+TCAP su DVS-Gesture seed 42. `replicate` cambia soltanto seed, ordine dei batch e
inizializzazione, poi profila il best senza ripetere il bounded-overfit già superato.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-f-tcap-43 -- replicate --config configs/dvslip_f_temporal_capacity.yaml --seed 43
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-f-tcap-44 -- replicate --config configs/dvslip_f_temporal_capacity.yaml --seed 44
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvsgesture-f-tcap-42 -- candidate --config configs/dvsgesture_f_temporal_capacity.yaml
```

Il run d8 rimane autorizzato come ultimo probe architetturale e può partire quando si libera un
server:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-f-tcap-d8-42 -- candidate --config configs/dvslip_f_temporal_capacity_d8.yaml
```

La baseline di confronto è `dvsgesture_e0__20260825_213021__seed42`: 84,47% accuracy e 83,62%
Macro-F1. La config di trasferimento conserva split, E0, 100 step, recipe e assenza di flip; cambia
soltanto la topologia in F+TCAP.

## Probe high-frequency F+TCAP

Dopo il check sul commit pulito, avviare il candidato registrato:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-f-tcap-dwc3-42 -- candidate --config configs/dvslip_f_tcap_stage1_dwc3.yaml
```

Il workflow esegue bounded overfit, full da pesi nuovi e profiling v4. La config sostituisce soltanto
Token-QK nello stage 1 con il mixer depthwise 3×3; E0, F, TCAP 1/2/4, stage 2 e recipe restano
invariati. Non cambiare kernel, ritardi o soglia del gate.

## Monitoraggio e ripresa

```bash
screen -ls
tail -f artifacts/screen/dvslip-tcap-taps42.log
tail -f artifacts/screen/dvslip-f-tcap-dwc3-42.log
screen -r dvslip-f-tcap-dwc3-42
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
