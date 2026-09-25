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
parametri pubblicati `p=0,5`, `Beta(1,1)` e tre componenti GMM, completati dalla realizzazione GMM
full-resolution del codice BrainCog degli autori. Maskout applica otto intervalli da uno a cinque
bin DVS-Gesture, ossia 200–1.000 ms. Entrambi si confrontano con il frozen seed 42 senza
augmentation.

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

## Fase predictive-temporal

Programma e motivazioni: sezione 12 di
[`PREDICTIVE_TEMPORAL_PHASE1_AUDIT.md`](PREDICTIVE_TEMPORAL_PHASE1_AUDIT.md). La prima esecuzione è
archiviata sotto `artifacts/superseded/`; il report dell'audit è in
`artifacts/predictive_phase1_audit/phase1_audit.json` ed è richiesto da ogni ingresso di training.

**Un job per GPU.** I due fallimenti dell'audit del 23 settembre erano OOM causati da un processo
residuo della campagna precedente e da job concorrenti. Prima di ogni lancio `nvidia-smi` deve
mostrare la GPU libera; `run_command.sh` rifiuta inoltre un worktree non pulito.

### 1. Obbligatorio: rigenerare l'audit con A1 stratificato

La versione registrata di A1 misurava 16 campioni di una sola parola. Il contratto corrente accetta
soltanto lo schema 2: quattro batch stratificati su 64 classi e A2 con 8192/2048 campioni. Il report
presente, schema 1, blocca intenzionalmente ogni ingresso di training finché questo comando non lo
rigenera.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-predictive-phase1-audit -- \
  predictive-phase1-audit \
  --c0-config     artifacts/dvslip_f_tcap_stage1_dwc3_d8__20260914_093427_434930__seed42/config_resolved.yaml \
  --c0-checkpoint checkpoints/dvslip_f_tcap_stage1_dwc3_d8__20260914_093427_434930__seed42/best.pt \
  --r0-config     artifacts/superseded/dvslip_predictive_r0_discriminative_lr__20260921_110121_131705__seed42/config_resolved.yaml \
  --r0-checkpoint checkpoints/dvslip_predictive_r0_discriminative_lr__20260921_110121_131705__seed42/best.pt \
  --s0-config     artifacts/superseded/dvslip_predictive_s0__20260922_210709_490368__seed42/config_resolved.yaml \
  --s0-checkpoint checkpoints/dvslip_predictive_s0__20260922_210709_490368__seed42/best.pt \
  --output        artifacts/predictive_phase1_audit \
  --fit-samples 8192 --holdout-samples 2048 --feature-samples 256
```

### 2. Preflight da leggere prima di ogni lancio

Il preflight è eseguito anche dai workflow, ma lanciarlo da solo permette di leggere il report prima
che parta il training. Campi da leggere: `initialization_passed`, `causal_prefix_passed`,
`diagnostic_batch_classes` (16), `shared_gradient_ratio` rispetto a `minimum_shared_gradient_ratio`,
`authority_calibration`, e `training_batchnorm_prefix_max_abs_difference`, che per i bracci da
zero è atteso diverso da zero e va solo registrato.

```bash
for arm in r0 late_prefix dynamic_tcap s0 s1; do
  CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh --foreground \
    predictive-check --config configs/dvslip_predictive_$arm.yaml \
    --output artifacts/predictive_preflight/dvslip_predictive_${arm}__seed42.json
done
```

### 3. Bracci del seed 42

Continuazioni da C0 (L15 e il suo controllo R0) e bracci da zero con la ricetta di C0. I server sono
macchine fisiche distinte: su ciascuna si usa la GPU locale `0`. Con quattro server, la prima ondata
contiene R0, L15, D e S0; S1 parte sulla prima macchina che si libera, dopo che S0 ha prodotto il
proprio controllo diretto **e** ha mostrato una skill predittiva di validation finita e positiva
rispetto ai riferimenti causali. Se S0 non apprende il meccanismo, S1 non riceve un full run.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-predictive-r0 -- predictive-continuation --config configs/dvslip_predictive_r0.yaml
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-predictive-late-prefix -- predictive-continuation --config configs/dvslip_predictive_late_prefix.yaml
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-predictive-dynamic-tcap -- predictive-scratch --config configs/dvslip_predictive_dynamic_tcap.yaml
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-predictive-s0 -- predictive-scratch --config configs/dvslip_predictive_s0.yaml
```

Seconda ondata, soltanto dopo la lettura della firma meccanicistica di S0:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-predictive-s1 -- predictive-scratch --config configs/dvslip_predictive_s1.yaml
```

**Stato al 2026-09-26.** L15 è completato. D va rieseguito con lo stesso comando: il primo run
aveva l'ampiezza esponenziale difettosa. S0 ha fallito il gate soltanto sulla CE, ma il full resta
sospeso fino al controllo held-out della decisione 12. Prima si archiviano il D difettoso e il gate
S0 duplicato, interrotto all'epoca 60:

```bash
mkdir -p artifacts/superseded
mv artifacts/dvslip_predictive_dynamic_tcap_overfit__20260925_103713_162684__seed42 \
   artifacts/dvslip_predictive_dynamic_tcap__20260925_104725_998808__seed42 \
   artifacts/dvslip_predictive_s0_overfit__20260925_200905_695697__seed42 \
   artifacts/superseded/
```

Solo dopo il GO del controllo held-out, S0 e il profiling del suo checkpoint deployabile sono:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-predictive-s0 -- train --config configs/dvslip_predictive_s0.yaml
# a run concluso, con <run-id> = dvslip_predictive_s0__<timestamp>__seed42
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-predictive-s0-profile -- profile-checkpoint --config artifacts/<run-id>/deployment_config_resolved.yaml --checkpoint checkpoints/<run-id>/deployment.pt --output artifacts/<run-id>/hardware_profile_v4.json --samples 64
```

Il primo controllo si fa dopo l'epoca 1: con il ramp a peso zero deve coincidere con C0. I valori di
riferimento sono nella sezione 12.9 dell'audit.

Le due diagnostiche checkpoint-only rimaste aperte usano lo stesso comando. Il primo job produce
A4 per L15; il secondo fornisce il riferimento `V_Δ` di C0 (e anche il suo A4 nello stesso formato):

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh predictive-a4-l15 -- \
  predictive-checkpoint-audit \
  --config artifacts/dvslip_predictive_late_prefix__20260924_153414_101029__seed42/deployment_config_resolved.yaml \
  --checkpoint checkpoints/dvslip_predictive_late_prefix__20260924_153414_101029__seed42/deployment.pt \
  --output artifacts/dvslip_predictive_late_prefix__20260924_153414_101029__seed42/checkpoint_audit

CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh predictive-vdelta-c0 -- \
  predictive-checkpoint-audit \
  --config artifacts/dvslip_f_tcap_stage1_dwc3_d8__20260914_093427_434930__seed42/config_resolved.yaml \
  --checkpoint checkpoints/dvslip_f_tcap_stage1_dwc3_d8__20260914_093427_434930__seed42/best.pt \
  --output artifacts/dvslip_f_tcap_stage1_dwc3_d8__20260914_093427_434930__seed42/checkpoint_audit
```

Ogni output contiene `predictive_checkpoint_audit.json` e `a4_per_sample.csv`. Il JSON riporta
`temporal_variation_stage1_active` e `temporal_variation_stage2_active` sotto
`A4_tail_margin.temporal_variation`. Se il checkpoint contiene il predittore S0, la sezione
`A4_tail_margin.temporal_diagnostics` riporta anche loss e skill contro persistenza e media dei
ritardi sull'intera development-validation.

Prima di autorizzare il full S0, il controllo held-out del checkpoint del gate è:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh predictive-s0-heldout-audit -- \
  predictive-checkpoint-audit \
  --config artifacts/dvslip_predictive_s0_overfit__20260925_154302_017223__seed42/config_resolved.yaml \
  --checkpoint checkpoints/dvslip_predictive_s0_overfit__20260925_154302_017223__seed42/best.pt \
  --output artifacts/dvslip_predictive_s0_overfit__20260925_154302_017223__seed42/heldout_checkpoint_audit
```

Questo job non riaddestra il modello. Per S0 vanno interpretate le metriche predittive; accuracy e
F1 del classificatore non sono un test utile, perché il checkpoint è stato addestrato su 64 sample.

Ogni workflow esegue preflight, gate bounded (finestra finale a peso ausiliario pieno), run completo
e profiling del checkpoint deployabile, e scrive `predictive_workflow.json`. Nella storia per epoca
di S0 e S1 vanno letti `auxiliary_nominal_weight`, `authority_nominal_shared_ratio`,
`authority_shared_ratio` effettivo dopo il ramp, `authority_shared_cosine` e
`train_temporal_variation_stage{1,2}_active`.

### 4. Seed 43 e 44, solo per i bracci con firma coerente

I bracci da zero si replicano con la stessa configurazione; i loro controlli sono i seed 43 e 44
archiviati di C0.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh dvslip-predictive-s0-43 -- replicate --config configs/dvslip_predictive_s0.yaml --seed 43
```

Le continuazioni non si replicano così: un seed diverso richiede il C0 dello stesso seed come parent
e come teacher, e il gate dell'audit confronta l'hash del parent con il C0 esaminato dall'audit
(seed 42). Va deciso prima, se servirà.

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
