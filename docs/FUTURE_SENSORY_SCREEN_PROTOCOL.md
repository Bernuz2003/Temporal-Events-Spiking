# Screen del futuro sensoriale su DVS-Lip

**Definito il 1 ottobre 2026; codice e controlli ancora da realizzare.**
Lo screen sceglie un *obiettivo di pretraining*, non una nuova rappresentazione di input né un
nuovo modulo da mantenere in deployment. Usa soltanto il development-train; la development
validation e l'official test restano chiusi fino alla decisione sul candidato completo.

## Domanda e motivazione

S0 e il probe dell'innovazione mostrano che il latente di D è prevedibile, ma rendere prevedibile
quel latente non ha dato un guadagno classificativo attribuibile alla predizione. Il nuovo test
chiede invece se **un passato causale degli eventi può anticipare una proprietà spaziale e
temporale misurabile degli eventi futuri**, e se addestrarsi a farlo organizza una rappresentazione
più utile alla classe rispetto a una ricostruzione del presente. Il successo di D e TCAP rende
plausibile la disponibilità di storia; il settling post-evento di D e PLIF impone di non lasciare
che la coda vuota domini lo screen. Il fallimento del precedente `fine_future` riguardava un
*latent di un teacher MG* poco accessibile dal contesto studente; non falsifica questo target
fisico, ma impone di dimostrarne la predicibilità anziché presumerla.

La predizione di eventi futuri come fonte di rappresentazioni è motivata da
[Fast Feature Field](https://arxiv.org/abs/2509.25146); la costruzione di campi evento con
kernel temporali e voxel è inquadrata da
[Gehrig et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Gehrig_End-to-End_Learning_of_Representations_for_Asynchronous_Event-Based_Data_ICCV_2019_paper.html).
**Nessuno dei due lavori dimostra che FEPF-2 migliori DVS-Lip**: il target a momenti qui sotto è
una nostra ipotesi da verificare.

## Variabile predetta, fissata prima di vedere gli esiti

Il contesto principale è E0 originale (`40 × 2 × 128 × 128`, bin fisici di 50 ms), fino al cutoff
`t`: soltanto eventi con timestamp `< t`. Non si forniscono al modello durata finale, ultimo
evento, feature di D o etichette. Il futuro è l'intervallo semiaperto `[t,t+H)`, con **H = 100 ms**
e cutoff ogni 50 ms. `H` copre due bin E0: è abbastanza lungo da esporre una transizione breve,
senza assumere che un singolo micro-evento sia predicibile. Questo valore è un'ipotesi fissa,
non il primo punto di uno sweep. Si passa a `H` diverso soltanto con un nuovo protocollo.

Gli eventi futuri ON e OFF vengono accumulati, senza clipping, su una griglia fissa `16 × 16`
(celle sensore `8 × 8`), allineata allo stage1 di D. Questa è la risoluzione del *target*; il
contesto mantiene la risoluzione E0 e usa una riduzione spaziale appresa. Nessuna normalizzazione
della durata o del totale eventi dell'utterance entra nel target. Per cella `u`, polarità `p` ed
evento futuro `i`, sia `φ_i=(t_i-t)/H ∈ [0,1)`.

| Braccio | Target per `p,u` | Cosa conserva |
|---|---|---|
| **FEPF-2, primario** | `B0=Σ(1−φ)²`, `B1=Σ2φ(1−φ)`, `B2=Σφ²` | count, tempo medio e dispersione temporale; 6 canali ON/OFF |
| **Voxel-4, sfidante** | quattro count nei quarti consecutivi di `[t,t+H)` | ordine temporale grossolano in 4 intervalli di 25 ms; 8 canali ON/OFF |

In entrambi i casi `N=Σ_j target_j` è **esattamente** il count futuro per cella e polarità.
FEPF-2 consente `M1=B1/2+B2` e `M2=B2`; il voxel conserva invece la struttura per quarti.
Sono due discretizzazioni **non annidate**: il voxel hard non è un upper bound matematico dei
momenti continui, perché perde il timestamp entro ciascun quarto. `M0=count` è il controllo di
densità ottenuto gratuitamente per somma; **non è un terzo training dello screen**.

## Disegno del fit e controlli

1. Partizionare una volta i 11.901 utterance del development-train in fit/holdout `80/20`,
   stratificando per classe e registrando indici e hash. Lo split esistente non è speaker-disjoint
   perché l'identità speaker non è disponibile: questa limitazione va dichiarata. Niente
   development validation, official test o selezione di checkpoint tramite queste partizioni.
2. Misurare sul solo fit la distribuzione di count, celle vuote, finestre attive e durata. Un
   bounded overfit su campioni del fit deve verificare capacità del predittore e target non
   degeneri **prima** del fit principale. Se fallisce, si corregge l'implementazione o si
   ridisegna lo screen *prima* di aprire l'holdout; non si conclude che la prediction sia inutile.
3. Addestrare FEPF-2 e Voxel-4 da zero con **la stessa topologia del trunk D G=1**
   (F+DWC-3+TCAP-d8 e router dinamico di stage2, larghezza 128), con normalizzazione
   temporalmente causale e senza inizializzazione dal checkpoint
   supervisionato; stesso seed, input, ordine dei batch e decoder salvo il numero di canali
   d'uscita. Il decoder di solo training fonde la feature stage1 a `16 × 16` con quella stage2
   riportata alla stessa griglia: la loss deve attraversare entrambi gli stage e il router.
   Nessun limite artificiale di parametri/compute per la fase di scoperta; il costo è misurato.
   Budget iniziale comune: **40 epoche sul fit**. Se la loss di count o timing di almeno un
   braccio scende ancora di oltre il 5% nelle ultime 8 epoche, estendere **entrambi** a 80 prima di
   leggere l'holdout; non estendere un solo braccio in base al suo punteggio. Il braccio vincente
   riceve altre **due** inizializzazioni se il primo screen è positivo: tre seed in totale.
4. L'ultimo bin occupato definisce soltanto la **maschera della loss**: usare cutoff per cui
   `[t,t+H)` rientra nell'utterance attiva. Non escludere finestre senza eventi *all'interno*
   della regione attiva. Ogni utterance pesa uguale, indipendentemente da durata e densità.
   L'endpoint non è input del predittore né del classificatore. Riportare anche l'esito per
   fascia di durata, per verificare la selezione introdotta dalla maschera.
5. Predire separatamente intensità futura `N̂ ≥ 0` e distribuzione temporale condizionale
   `q̂=softmax(logits)` sulle tre o quattro componenti. Per celle con `N>0`, il target temporale
   è `q_j=target_j/N`. La loss comprende una devianza di count con celle occupate/vuote
   bilanciate e una KL `q || q̂` sulle celle occupate; le due parti sono normalizzate con scale
   determinate **dal solo fit** e registrate. Così il modello non può vincere predicendo sempre
   zero o sfruttando soltanto densità. Riportare le metriche grezze oltre a quelle normalizzate.
6. Baseline causali senza un nuovo modello: count zero come controllo di degenerazione, media
   del fit condizionata al cutoff e
   alla polarità, persistenza del count nell'ultima finestra di 100 ms; per `q`, prior del fit
   e profilo temporale della finestra passata (fallback al prior quando vuota). Separare skill
   del count e della KL temporale; riportare tutte le celle, sole celle occupate, regione
   attiva, attività ON/OFF e varianza delle predizioni. Una skill globale dominata dagli zeri
   non supera lo screen.
7. Il riferimento secondario con feature di D congelato può stimare un limite *ottimistico* di
   accessibilità, ma D è già supervisionato: **non può far superare il gate se il passato E0
   senza etichette fallisce**. Non introduce un teacher nel target.

La BatchNorm attuale del backbone appiattisce `T × B` prima della normalizzazione: durante il
training una feature al cutoff può quindi dipendere dal futuro del medesimo utterance. Lo screen
deve usare normalizzazione locale al campione e al passo (per esempio GroupNorm), oppure un
meccanismo causale dimostrato equivalente. Preflight obbligatorio in `train()` **e** `eval()`:
perturbare eventi dopo `t` non cambia rappresentazione, predizione o gradiente al cutoff; il
prefisso isolato coincide con il prefisso del forward completo entro tolleranza numerica.
BatchNorm che vede tutta la sequenza non è ammessa come prova di prediction causale.

## Misura dell'utilità e regola di decisione

La ricostruzione non basta. Sul holdout interno, senza riaddestrare l'encoder, si confrontano:

- **informazione temporale osservabile:** una stessa piccola sonda di classe sul futuro *vero*
  completo contro il suo `M0` soltanto, dichiarando esplicitamente che è un oracle non causale;
- **informazione temporalmente anticipata:** due sonde riaddestrate con stessa architettura,
  budget e seed sulle predizioni complete oppure su `N̂` soltanto (canali temporali sostituiti
  dal prior); non si misura soltanto il danno provocato alla testa completa togliendole i canali;
- **qualità della rappresentazione:** identica testa per passo temporale, logit mediati sui 40
  passi, su encoder SSL congelato rispetto a encoder casuale congelato. Riportare Macro-F1
  finale e ai prefissi fissi di 1,0/1,5 s. Le teste usano etichette del fit e sono valutate su
  utterance disgiunti; il readout non usa l'endpoint futuro.

Un braccio entra nella seconda verifica solo se, nel contesto E0, entrambe le componenti
**count e timing** migliorano almeno del **5% relativo** rispetto alla migliore baseline causale
pertinente sul holdout attivo, con limite inferiore bootstrap appaiato per utterance sopra zero.
La sonda sulle predizioni complete deve inoltre battere `N̂`-only di almeno **1 pp Macro-F1**,
con limite inferiore bootstrap sopra zero; la sonda dell'encoder deve battere di almeno **1 pp**
l'encoder casuale congelato. Queste sono soglie operative di ROI, non costanti teoriche.
Non si decide usando milioni di celle come repliche indipendenti: unità statistica e bootstrap
sono l'utterance. F1, differenze e curve dei singoli seed restano visibili.

Se il primo passaggio è positivo, il vincitore riceve due controlli **prima** del full:
(a) replica con altri due seed; (b) stesso encoder, testa e budget addestrati a ricostruire il
campo corrispondente nell'intervallo passato `[t−H,t)` anziché quello futuro. Questa seconda
prova separa la prediction dal beneficio generico di un pretraining su eventi. Promozione solo
se il vincitore conserva skill temporale e miglioramento della sonda dell'encoder in **tutti e
tre** i seed, e batte il controllo sul presente in Macro-F1 in almeno **2/3 coppie** con delta
medio di almeno **1 pp** sul holdout. Il bootstrap sui campioni non sostituisce la dispersione
fra i tre fit: si riportano entrambi. I numeri sono
segnali di selezione, non prova di guadagno end-to-end; nessun esito del checkpoint D seed 42
o dell'innovation probe può sostituirli.

Se entrambi i target passano, scegliere quello con migliore trasferimento della rappresentazione
e skill temporale stabile; in caso di risultati indistinguibili preferire FEPF-2 per la sua
semantica compatta, **non** per un vincolo di hardware anticipato. Non combinare i target in
questa fase. Se nessuno passa, non avviare un full predittivo; il risultato negativo vale per
questa interfaccia e questo orizzonte, non per ogni predictive coding.

## Solo dopo uno screen positivo

Un solo candidato (D o Groupwise-D, in base ai rispettivi risultati strutturali) viene
preaddestrato da zero a prevedere il target vincente, con decoder rimovibile e fine-tuning CE
puro. La loss SSL deve raggiungere anche lo stage2 e il router, non soltanto lo stem. Si usa la
stessa politica di normalizzazione causale nello screen, nel pretraining e nel controllo di
training da zero; altrimenti il cambio BN/GroupNorm confonde il confronto. Se il full migliora
Macro-F1 e prefissi rispetto al controllo appaiato, eseguire il full di ricostruzione del
presente a pari passi di ottimizzazione per attribuire il guadagno al futuro. Solo allora
replicare su altri seed e fare profiling del modello deployabile, senza decoder.
