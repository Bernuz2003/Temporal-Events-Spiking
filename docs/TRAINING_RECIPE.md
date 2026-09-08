# Ricetta di training congelata

**Aggiornata:** 2026-09-08

**Recipe ID base:** `dvslip_e0_128`

Durante la selezione architetturale si usano: AdamW, learning rate `3e-4`, minimo `1e-6`, cosine
decay per 128 epoche, warmup di 4 epoche da fattore `0.01`, weight decay `5e-4`, label smoothing
`0.1`, clipping globale `1.0`, accumulo di 2 batch e AMP su CUDA. Il best checkpoint è selezionato
per Macro-F1. Batch size, rappresentazione e horizontal flip restano quelli di
`configs/dvslip_e0.yaml`.

## Gate prima di un full run

1. config valida e differenza dalla baseline limitata ai campi architetturali dichiarati;
2. output `[B,100]`, loss e gradienti finiti;
3. test di causalità per ogni nuovo operatore temporale;
4. bounded overfit deterministico su 16 classi × 4 campioni, massimo 500 epoche / 1,000 optimizer
   step, senza augmentation e AMP: cinque epoche consecutive con accuracy ≥95%, loss <1.5 e
   gradienti finiti nell'intera storia; stop anticipato al superamento;
5. stima strutturale di parametri, stato e operazioni disponibile.

Lo smoke dimostra solo il funzionamento della pipeline. L'overfit individua errori di
inizializzazione o di flusso del gradiente; non predice la generalizzazione.

`etsr candidate --config ...` applica questi controlli numerici, salva `overfit_gate.json` e gli
indici del subset, blocca il full run in caso di fallimento e riparte da pesi nuovi dopo il successo.
La decisione usa le ultime cinque epoche, non il best selezionato. Il full run conserva la ricetta
base esatta e viene seguito automaticamente da `hardware_profile_v4.json` sul proprio best.
`candidate_workflow.json` collega gate, full run e profilo. Test di forma/causalità/CUDA sono
il controllo preliminare `dataset_workflow.sh dvslip check`, da eseguire una volta per commit.

## Modifiche vietate nella discovery

Non cambiare contemporaneamente optimizer, schedule, clipping, augmentation, binning o loss. Non
estendere le epoche oltre il cosine già esaurito. Un'eccezione richiede una decisione documentata e
un controllo capace di separarla dalla modifica architetturale.

## Fase post-freeze

Dopo la conferma multi-seed si crea un nuovo `recipe_id`. La prima verifica ammessa confronta la
stessa augmentation sulla baseline e sulla candidata congelata. Ulteriore tuning avviene in modo
sequenziale e si arresta appena il risultato non cambia la conclusione.
