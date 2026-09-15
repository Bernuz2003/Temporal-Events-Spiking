# Operazioni riproducibili su SMILIES

**Aggiornate:** 2026-09-15

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

Il run combinato DWC-3+d8 è il controllo diretto dell'ultimo probe a ritardi apprendibili. In parallelo si replicano le baseline B
ai seed 43/44, utili qualunque candidata venga selezionata. Il candidato passa prima dal bounded
overfit; `replicate` esegue il full da zero e profila il best senza ripetere il gate già validato.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-f-tcap-dwc3-d8-42 -- candidate --config configs/dvslip_f_tcap_stage1_dwc3_d8.yaml
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-b-43 -- replicate --config configs/dvslip_e0.yaml --seed 43
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-b-44 -- replicate --config configs/dvslip_e0.yaml --seed 44
```

Per l'unico probe DWC-3+TCAP con quattro ritardi apprendibili per canale, eseguire il check CUDA
prima del gate. Il best e il profilo usano i ritardi interi; il full parte soltanto se passa l'overfit.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/dataset_workflow.sh dvslip check
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-f-tcap-dwc3-learned-42 -- candidate --config configs/dvslip_f_tcap_stage1_dwc3_learnable_delays.yaml
```

Le repliche del finalista e il trasferimento DVS-Gesture seguono il confronto fra i candidati.

## Fase C1 — screening parallelo delle augmentation

Il primo raffinamento supervisionato applica soltanto al modello congelato seed 42 otto maschere
temporali lunghe da uno a quattro bin da 50 ms. Validation e profiling restano senza augmentation;
il confronto diretto è con lo stesso modello seed 42 già addestrato senza Maskout. Il secondo run
isola lo spatial erasing usato nella pipeline DVS-Lip pubblica, senza aggiungere altre trasformazioni.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-final-maskout-42 -- refine --config configs/dvslip_f_tcap_stage1_dwc3_d8_temporal_maskout.yaml
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-final-spatial-erasing-42 -- refine --config configs/dvslip_f_tcap_stage1_dwc3_d8_spatial_erasing.yaml
```

## Trasferimento strutturale su DVS-Gesture

Il primo confronto trasferisce a seed 42 l'intera architettura congelata F+DWC-3+TCAP-d8,
mantenendo il protocollo DVS-Gesture e i ritardi `[1,2,4,8]` in unità di bin. Il workflow esegue
bounded overfit, full training da pesi nuovi e profiling del best checkpoint.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvsgesture-f-tcap-dwc3-d8-42 -- candidate --config configs/dvsgesture_f_tcap_stage1_dwc3_d8.yaml
```

## Fase C — screening augmentation su DVS-Gesture

I due run isolano EventMix e Temporal Maskout sul medesimo finalista seed 42. EventMix usa i
parametri pubblicati `p=0,5`, `Beta(1,1)` e tre componenti GMM; le scelte non specificate dal paper
sono dichiarate nella config. Maskout applica otto intervalli da uno a cinque bin DVS-Gesture,
ossia 200–1.000 ms. Entrambi si confrontano con il frozen seed 42 senza augmentation.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvsgesture-final-eventmix-42 -- refine --config configs/dvsgesture_f_tcap_stage1_dwc3_d8_event_mix.yaml
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvsgesture-final-maskout-42 -- refine --config configs/dvsgesture_f_tcap_stage1_dwc3_d8_temporal_maskout.yaml
```

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
