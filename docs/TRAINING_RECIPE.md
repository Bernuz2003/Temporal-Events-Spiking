# Ricetta di training congelata

**Aggiornata:** 2026-09-19

**Recipe ID base:** `dvslip_e0_128`

Durante la selezione architetturale si usano: AdamW, learning rate `3e-4`, minimo `1e-6`, cosine
decay per 128 epoche, warmup di 4 epoche da fattore `0.01`, weight decay `5e-4`, label smoothing
`0.1`, clipping globale `1.0`, accumulo di 2 batch e AMP su CUDA. Il best checkpoint è selezionato
per Macro-F1. Batch size e horizontal flip restano quelli di `configs/dvslip_e0.yaml`. Le
rappresentazioni TBR usano il backbone F invariato salvo `in_channels=1`, mantengono 40 macro-step
e fissano senza sweep `N=8`, `Δt=6,25 ms`; Spike-TBR fissa inoltre `β=0,9` e soglia `1,1`.

## Gate prima di un full run

1. config valida e differenza dalla baseline limitata ai campi architetturali dichiarati;
2. output `[B,100]`, loss e gradienti finiti;
3. test di causalità per ogni nuovo operatore temporale;
4. bounded overfit deterministico su 16 classi × 4 campioni, massimo 500 epoche / 1,000 optimizer
   step, senza augmentation e AMP: cinque epoche consecutive con accuracy ≥95%, loss <1.5 e
   gradienti finiti nell'intera storia; stop anticipato al superamento;
5. stima strutturale di parametri, stato e operazioni disponibile;
6. per una nuova rappresentazione, test dell'invariante dichiarato (count oppure occupazione/bit),
   tempo fisico, assenza di endpoint oracle e compatibilità con augmentation.

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

**Eccezione preregistrata 2026-09-19:** prima di riprendere augmentation si applicherà il
protocollo `dvslip_predictive_continuation_32_v1` definito nella
[roadmap predittiva](PREDICTIVE_TEMPORAL_ROADMAP.md). È una continuazione di C0 con controllo
appaiato, BN running fissa e recipe comune, non un nuovo full `dvslip_e0_128`. Il workflow
`predictive-continuation` applica il preflight causale/gradienti, il gate bounded e il profiling
del checkpoint deployabile. CE e loss ausiliarie sono registrate separatamente; i pesi allenati
nel gate non entrano nel run di produzione.

Dopo la conferma multi-seed si crea un nuovo `recipe_id`. Ogni raffinamento viene applicato alla
sola candidata congelata e confrontato con il run della stessa struttura e dello stesso seed senza
la modifica; B non viene riaddestrata. Ulteriore tuning avviene in modo sequenziale e si arresta
appena il risultato non cambia la conclusione.

Il primo screen usa `recipe_id=dvslip_e0_128_tm8x4`: otto maschere temporali indipendenti, ciascuna
lunga uniformemente da uno a quattro bin da 50 ms. Le maschere possono sovrapporsi e sono applicate
solo al training; validation, curve a prefisso e profiling ricevono l'input integro. La relativa
config seleziona il modello congelato seed 42. Il workflow `refine` è comune ai dataset: legge dalla
config il riferimento, verifica che ogni altra sezione sia invariata e profila il best.


## Diagnostiche senza training

`temporal-diagnostic-pair` carica esclusivamente best checkpoint compatibili e non modifica la
ricetta. Le valutazioni a prefisso sono esplicitamente fuori dall'orizzonte di training fisso; le
curve event-aligned usano endpoint oracle. Questi output possono motivare un'ipotesi futura ma non
possono selezionare una costante di settling o autorizzare da soli un full.
