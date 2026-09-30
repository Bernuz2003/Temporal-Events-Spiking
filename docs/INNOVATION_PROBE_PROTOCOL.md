# Probe dell'innovazione latente su D congelato

**Protocollo fissato il 30 settembre 2026.** Questo è uno screen *checkpoint-only* su
DVS-Lip development, non un nuovo training del backbone. D è il best checkpoint seed 42 di
`dvslip_predictive_dynamic_tcap__20260925_225604_461593__seed42` (58,31% Macro-F1 sulla
validation). L'official test resta escluso.

## Domanda e interpretazione

L'ipotesi è che un osservatore della storia, addestrato a prevedere il latente presente senza
etichette, organizzi la memoria temporale in modo più utile alla classificazione di quanto faccia
lo stesso osservatore addestrato soltanto con la cross-entropy. Non si sostiene che l'osservatore
crei informazione: la sua uscita è una funzione deterministica della storia disponibile al
controllo. Un eventuale vantaggio riguarda *rappresentazione e apprendimento* a capacità finita.

La sonda è `z_t`, uscita dello stage2 dopo media spaziale, `[40,128]` per enunciato. D è congelato
e in `eval()` con BatchNorm a statistiche fisse. Non si cambia encoder E0, split, checkpoint o
readout `mean + fixed_window`. Questa posizione è esattamente quella che precede la media
temporale e la testa lineare di D. Un esito negativo chiude l'ipotesi soltanto a questa interfaccia
di lettura, non all'interno di stage1/stage2 o per un backbone riaddestrato.

| Braccio | Ingresso e percorso | Cosa distingue |
|---|---|---|
| D esatto | testa originale di D su `mean_t(z_t)` | controllo: logit identici a D |
| Lineare riaddestrata | nuova testa lineare su `z_t`, media dei logit | sensibilità alla riottimizzazione su feature finali congelate; il divario da D non è un bias da sottrarre |
| **C** | storia `[z_{t-1},z_{t-2},z_{t-4},z_{t-8}]` → Q `512→256→128`, concatenazione con `z_t`, testa `256→256→100` | Q pre-addestrato a predire `z_t`, poi congelato |
| **B** | topologia, inizializzazione, parametri e percorso di C | codificatore della storia addestrato da zero soltanto con la loss di classe |
| **B_rand** | topologia e inizializzazione di C | codificatore casuale congelato; isola il solo effetto del congelamento |
| A, B piatto | `z_t→128→256→100`; storia grezza `640→345→100` | descrittivi: stato e decoder diretto della storia |
| R_Q, R_P | `z_t−Q(storia)`; `z_t−z_{t-1}`, ciascuno `128→256→100` | innovazione latente appresa contro differenza latente rispetto alla persistenza; non replica del sensore DVS |

I tre sistemi principali B, B_rand e C hanno ciascuno **255.716 parametri totali**; C e B_rand
ne aggiornano 91.492 durante la classificazione, B tutti. Il pretraining aggiunge passi di
ottimizzazione a C: è parte dell'intervento, non un costo nascosto. Tutti i classificatori
applicano il loro MLP *a ciascun passo* e ottimizzano la cross-entropy **dopo** la media dei 40
logit. Non impongono la classe a ogni istante. Le quattro posizioni storiche mancanti sono zero
per tutti i bracci; non si usa endpoint futuro. Media e deviazione standard dei canali derivano
soltanto dal development-train. L'ancora esatta usa le feature non normalizzate.

Q viene addestrato sui **40 passi** perché il readout di D utilizza anche la coda post-evento.
Addestrarlo solo sulla regione attiva e usare poi le sue uscite sulla coda creerebbe un cambio di
distribuzione. La skill è comunque riportata separatamente per regione attiva e coda; l'ultimo
bin con eventi definisce solo la *metrica*, non l'ingresso di Q o delle teste.

## Sequenza preregistrata

1. Il comando `innovation-probe-fit` legge **solo il development-train** senza augmentation.
   Estrae le feature con D congelato e verifica su un batch l'identità dei logit dell'ancora e
   l'invarianza causale delle feature ai prefissi. Per ciascuno dei seed `42–46`, inizializza
   identicamente gli encoder e le teste di B/B_rand/C, e usa lo stesso ordine dei minibatch
   nella fase di classificazione.
2. Q usa un fit/holdout casuale `80/20` interno al development-train. Budget fisso: **30 epoche
   Q**, poi **40 epoche** per ciascuna testa, batch 256, AdamW, learning rate `1e-3`, weight decay
   `1e-4`, senza early stopping. B, C e B_rand vedono tutte le etichette di development-train
   nella fase di classificazione. Il fit salva `training_curves.csv`, `fit.pt` e
   `fit_report.json`; nessun punteggio di validation è disponibile a questo punto.
3. Prima della validation, controllare le curve per escludere un B palesemente non convergente.
   Il gate meccanico di Q richiede in **almeno 4/5 seed**, sull'holdout interno e nella regione
   attiva: MSE inferiore sia alla persistenza sia alla media dei quattro ritardi, e rapporto fra
   deviazioni standard dell'uscita e del target ≥0,1 (calcolate per canale sui passi validi).
   Se fallisce, `innovation-probe-evaluate` si arresta senza aprire la
   validation. Questo holdout è nuovo per Q ma **non** per D, che ha già usato le sue etichette;
   non misura generalizzazione del backbone.
4. Dopo il fit, `innovation-probe-evaluate` legge la development validation **una sola volta**,
   ricostruisce tutte le teste dai pesi salvati e scrive `evaluation.json` e le predizioni. Riporta
   Macro-F1, accuracy, Acc1/Acc2, errori sulla parola gemella, prefissi 1,0/1,5 s e skill Q in
   validation. Un secondo lancio nella stessa directory è impedito.

Il confronto **primario** è C−B nel Macro-F1 finale, per *coppia* di seed. La condizione di
attribuzione è C−B_rand. Per ciascuno dei due confronti servono **entrambe**:

- media dei cinque delta positiva, con limite inferiore dell'intervallo t bilaterale 95%
  (`df=4`) strettamente maggiore di zero;
- almeno quattro delta su cinque strettamente positivi.

La regola è deliberatamente severa: un effetto piccolo o instabile non giustifica un training
completo. La condizione deve passare per **entrambi** i controlli. Il test t su soli cinque seed
è una regola di selezione operativa, non una garanzia distributiva; si riportano sempre i cinque
delta e la loro deviazione standard. Per ogni coppia sono aggiunti bootstrap appaiato e
stratificato per classe sugli enunciati (incertezza *condizionata* a quelle teste) e McNemar
esatto sull'accuracy. Non si moltiplicano artificialmente gli enunciati per cinque. La media dei
logit dei cinque seed è solo un risultato descrittivo: rappresenta un ensemble, non il modello
singolo che verrebbe eventualmente addestrato.

Un pass autorizza **un solo run end-to-end** del candidato; non dimostra un guadagno del backbone
riaddestrato. In caso contrario, si chiude questa proposta sulle feature finali del D attuale.
Il checkpoint D è stato selezionato su questa validation: il risultato resta esplorativo di
development, da confermare in seguito senza usare l'official test per selezionare varianti.

## Comandi SMILIES

Usare il commit sincronizzato e il checkpoint `best.pt` dello stesso D seed 42. Il wrapper
richiede un worktree pulito.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh innovation-fit-d42 -- \
  innovation-probe-fit \
  --config configs/dvslip_predictive_dynamic_tcap.yaml \
  --checkpoint checkpoints/dvslip_predictive_dynamic_tcap__20260925_225604_461593__seed42/best.pt \
  --output artifacts/innovation_probe_d42_fit
```

Solo dopo aver letto il report del fit e verificato il gate di Q e le curve:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smilies/run_command.sh innovation-eval-d42 -- \
  innovation-probe-evaluate \
  --config configs/dvslip_predictive_dynamic_tcap.yaml \
  --checkpoint checkpoints/dvslip_predictive_dynamic_tcap__20260925_225604_461593__seed42/best.pt \
  --fit-dir artifacts/innovation_probe_d42_fit \
  --output artifacts/innovation_probe_d42_evaluation
```
