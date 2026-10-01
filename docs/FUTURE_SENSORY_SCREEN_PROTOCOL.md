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
più utile alla classe rispetto a controlli di pretraining non futuri. Il successo di D e TCAP rende
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
   d'uscita. Il decoder di solo training legge **soltanto stage2** (`8 × 8`), ne porta la feature
   a `16 × 16` con upsampling appreso e produce il campo futuro. Stage1 riceve comunque il
   gradiente attraverso stage2; nessuno skip stage1→decoder può aggirare lo stage. Si registrano
   separatamente norme dei gradienti e aggiornamenti del router, senza dedurre utilità causale
   dal solo fatto che siano non nulli.
   Nessun limite artificiale di parametri/compute per la fase di scoperta; il costo è misurato.
   Budget iniziale comune: **40 epoche sul fit**. Se la loss di count o timing di almeno un
   braccio scende ancora di oltre il 5% nelle ultime 8 epoche, estendere **entrambi** a 80 prima di
   leggere l'holdout; non estendere un solo braccio in base al suo punteggio. Il braccio vincente
   riceve altre **due** inizializzazioni se il primo screen è positivo: tre seed in totale.
4. Il timestamp dell'ultimo evento `t_last` definisce soltanto la **maschera della loss**:
   sono validi i cutoff `0 < t < t_last` con `t+H ≤ 2 s`. La finestra futura può oltrepassare
   `t_last`: quegli zeri rappresentano l'offset reale dell'attività e conservano le transizioni
   finali, mentre i numerosi cutoff con `t ≥ t_last` non entrano nella loss. Non escludere
   finestre senza eventi *all'interno* della regione così definita. Ogni utterance pesa uguale,
   indipendentemente da durata e densità. L'endpoint non è input del predittore né del
   classificatore. Riportare separatamente finestre interamente attive e finestre che includono
   l'offset, oltre alla fascia di durata, per misurare l'effetto della selezione. Per i controlli
   sul passato, gli istanti precedenti a zero sono riempiti con zero come nello stato iniziale
   del modello: così anche l'onset resta nello screen.
5. Predire separatamente intensità futura `N̂=softplus(a)+10⁻⁶` e distribuzione temporale
   condizionale `q̂=softmax(logits)` sulle tre o quattro componenti. Per celle con `N>0`, il
   target è `q_j=target_j/N`. La devianza di count è
   `D(N,N̂)=2[N̂−N+N log(N/N̂)]`, con `D(0,N̂)=2N̂`; il termine temporale è
   `KL(q||q̂)=Σ_{j:q_j>0}q_j log(q_j/q̂_j)`. Per ogni utterance, mediare `D` separatamente su
   celle occupate e vuote, poi dare peso `1/2` a ciascun gruppo presente; mediare la KL sulle
   sole celle occupate; se un cutoff non ne ha, contribuisce solo alla loss di count. Mediare
   prima sui cutoff validi dell'utterance e poi sulle utterance:
   nessun campione pesa di più perché più lungo o denso. Siano `s_N` la devianza bilanciata
   della mean-field baseline sul **fit** e `s_q` la KL del prior temporale sul **fit**; la loss
   fissa è `0,5·D_bal/max(s_N,10⁻⁶) + 0,5·KL/max(s_q,10⁻⁶)`. Se una scala è degenere, il
   preflight si ferma prima dell'holdout. Riportare anche devianza e KL non normalizzate.
6. Baseline causali senza un nuovo modello: count zero come controllo di degenerazione;
   **mean field spaziale** `N̄_{t,p,u}` dal solo fit; persistenza del count nell'ultima finestra
   E0 di 100 ms; media dei count delle **ultime due finestre E0 di 100 ms** (200 ms totali).
   Per `q`, prior `q̄_{t,p,u}` calcolato sulle celle occupate del fit con uno pseudo-count per
   componente, più profilo della finestra passata e media dei due profili passati. I profili
   causali usano **solo E0**: per FEPF-2 gli eventi di ciascun bin E0 sono posti al suo centro
   temporale (fasi `0,25` e `0,75`); per Voxel-4 ogni count E0 è ripartito uniformemente nei
   due quarti corrispondenti. Quando il passato è vuoto, usare il prior del fit. Qualunque
   profilo calcolato da timestamp grezzi passati è un riferimento *privilegiato* separato,
   mai una baseline del gate E0-only. Separare skill del count e della KL temporale; riportare
   tutte le celle, sole celle occupate, finestre interne/di offset, ON/OFF e varianza delle
   predizioni. Una skill globale dominata dagli zeri non supera lo screen.
7. Il riferimento secondario con feature di D congelato può stimare un limite *ottimistico* di
   accessibilità, ma D è già supervisionato: **non può far superare il gate se il passato E0
   senza etichette fallisce**. Non introduce un teacher nel target.

La BatchNorm attuale del backbone appiattisce `T × B` prima della normalizzazione: durante il
training una feature al cutoff può quindi dipendere dal futuro del medesimo utterance. Per lo
screen fissiamo **GroupNorm applicata separatamente a ciascun passo e campione** al posto delle
BN del trunk (`4` gruppi per 16 canali, `8` gruppi da 32 canali in su); gli affine restano
addestrabili. Questo cambia il modello rispetto al D originale: il confronto BN-D resta un
benchmark prestazionale, **non** il controllo causale dell'effetto del pretraining. Preflight
obbligatorio in `train()` **e** `eval()`:
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

Per le sonde dei **campi futuri**, usare i cutoff fissi `t=50,100,...,750 ms` per tutti gli
utterance, senza selezionarli dall'ultimo evento; riportare anche la quota di finestre già
terminate. Il confronto sul futuro vero è un oracle diagnostico, mentre quello sulle predizioni
misura accessibilità precoce. Per la sonda dell'**encoder** si mantengono tutti i 40 passi e il
readout fisso di D, perché è questa rappresentazione che il full dovrà trasferire. Ogni sonda
dei campi riceve canali `[N,q]` per polarità; il controllo density-only mantiene la stessa shape
e sostituisce `q` con il prior del fit. Architettura fissata: due convoluzioni `3×3` a 128 canali,
GroupNorm locale al passo, GELU, media spaziale, lineare `128→100` per passo e media dei logit
sui cutoff fissati. La sonda dell'encoder usa la medesima testa con proiezione iniziale da 128
canali. Ciascun confronto ha stesso seed, ordine dei batch e **30 epoche fisse** di AdamW
(`lr=10⁻³`, `weight_decay=10⁻⁴`), senza augmentation e senza selezione sull'holdout; i parametri
effettivi di ogni testa vengono riportati. La capacità non è resa esattamente identica fra
target a 3/4 componenti, quindi il loro confronto resta una selezione operativa, non una prova
isolata dell'effetto della codifica.

Un braccio entra nella seconda verifica solo se, nel contesto E0, entrambe le componenti
**count e timing** migliorano almeno del **5% relativo** rispetto alla migliore baseline causale
E0-only sul holdout attivo. Formalmente `S_N=1−D_model/min_b D_b` e
`S_q=1−KL_model/min_b KL_b`, dove tutte le perdite sono prima mediate per utterance e `b`
varia sulle mean/persistence/recent-history pertinenti (il count zero è solo un allarme di
degenerazione). Si richiede anche un limite inferiore bootstrap appaiato per utterance sopra
zero per il guadagno assoluto rispetto a **ciascuna** baseline obbligatoria.
La sonda sulle predizioni complete deve inoltre battere `N̂`-only di almeno **1 pp Macro-F1**,
con limite inferiore bootstrap sopra zero; la sonda dell'encoder deve battere di almeno **1 pp**
l'encoder casuale congelato. Queste sono soglie operative di ROI, non costanti teoriche.
Non si decide usando milioni di celle come repliche indipendenti: unità statistica e bootstrap
sono l'utterance. F1, differenze e curve dei singoli seed restano visibili.

Se il primo passaggio è positivo, il vincitore riceve due controlli **prima** del full:
(a) replica con altri due seed; (b) stesso encoder, testa e budget addestrati a inferire il
campo **fine del passato** `[t−H,t)` anziché il futuro. Questo è un *matched non-future target*:
mantiene tipo di target e decoder, ma E0 non contiene timestamp intra-bin, quindi **non** è
ricostruzione esatta né isola da solo il beneficio di un generico pretraining. La ricostruzione
esatta delle ultime due count-frame E0, già contenute nell'input e aggregate sulla stessa griglia
`16 × 16`, è un controllo distinto di pretraining generico. Il suo target sono i quattro canali
ON/OFF dei due bin fisici da 50 ms in `[t−H,t)`, con zero-padding all'inizio; usa lo stesso
trunk, il decoder stage2-only, il budget di pretraining e una loss count normalizzata sul fit.
Non riceve subito un full; entra
nella verifica sequenziale soltanto se il full predittivo supera il controllo scratch.
Promozione solo se il vincitore conserva skill temporale e miglioramento della sonda dell'encoder in **tutti e
tre** i seed, e batte il controllo sul presente in Macro-F1 in almeno **2/3 coppie** con delta
medio di almeno **1 pp** sul holdout. Il bootstrap sui campioni non sostituisce la dispersione
fra i tre fit: si riportano entrambi. I numeri sono segnali di selezione, non prova di guadagno
end-to-end; nessun esito del checkpoint D seed 42
o dell'innovation probe può sostituirli.

Se entrambi i target passano, scegliere quello con migliore trasferimento della rappresentazione
e skill temporale stabile; in caso di risultati indistinguibili preferire FEPF-2 per la sua
semantica compatta, **non** per un vincolo di hardware anticipato. Non combinare i target in
questa fase. Se nessuno passa, non avviare un full predittivo; il risultato negativo vale per
questa interfaccia, l'orizzonte di 100 ms, la griglia `16 × 16` e il contesto E0, non per ogni
predictive coding.

## Solo dopo uno screen positivo

Un solo candidato (D o Groupwise-D, in base ai rispettivi risultati strutturali) viene
preaddestrato da zero a prevedere il target vincente, con decoder rimovibile e fine-tuning CE
puro. La loss SSL deve raggiungere stage2 e il router, non soltanto lo stem. Il **confronto
primario end-to-end** è appaiato fra:

1. `D_causalnorm_scratch`: stesso trunk, GroupNorm e training CE da zero, senza pretraining;
2. `D_causalnorm_future`: stesso trunk e GroupNorm, pretraining futuro seguito da fine-tuning CE.

Se vince Groupwise-D, sostituire entrambi con `G4_causalnorm` mantenendo l'appaiamento. Il
contrasto con il D/Groupwise-D originale a BatchNorm è soltanto prestazionale: un eventuale
vantaggio lì potrebbe derivare anche dalla normalizzazione. Solo se il full futuro batte il
controllo scratch, eseguire a pari passi di ottimizzazione il full *matched non-future* sui
campi fini del passato; se il segnale resta, eseguire il full di ricostruzione esatta E0.
Questi due controlli rispondono a domande differenti e nessuno, da solo, è una prova perfetta
che il futuro sia l'unica causa del guadagno. Soltanto dopo questa sequenza fare seed appaiati
e profiling del modello deployabile, senza decoder.
