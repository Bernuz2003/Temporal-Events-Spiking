# Audit della prima esecuzione di Predictive-Temporal-Coding

**Data:** 2026-09-23. **Aggiornato:** 2026-09-24 con l'esito dell'audit (sezione 12).
**Stato:** contratto attivo, vincolante per la riesecuzione della fase.

Questo documento registra che cosa, nella prima esecuzione della fase predittiva, è stato
implementato male, misurato male o formulato male. Non ridefinisce le ipotesi scientifiche: quelle
restano in [`PREDICTIVE_TEMPORAL_ROADMAP.md`](PREDICTIVE_TEMPORAL_ROADMAP.md) e la loro valutazione
critica in [`PREDICTIVE_TEMPORAL_RESEARCH_REVIEW.md`](PREDICTIVE_TEMPORAL_RESEARCH_REVIEW.md).

Definisce invece tre cose: quali risultati della prima esecuzione **cessano di valere come
evidenza**, quali **correzioni** devono essere implementate prima di rilanciare, e **come vengono
giudicati** i risultati della seconda esecuzione.

La fase viene rieseguita **sovrascrivendo** configurazioni e nomi di esperimento esistenti. Non si
creano varianti `_v2`, `_fixed` o `_corrected`. Le conseguenze di questa scelta sono dichiarate
nella sezione 10.

---

## 1. Verdetto sulla prima esecuzione

Sette continuazioni da 64 epoche, circa 47 GPU-ore, contro un budget pianificato di 44,86 e un
tetto invalicabile di 53,83. Nessun braccio ha superato il gate preregistrato di +1 pp.

Il motivo per cui questi run non valgono come evidenza sulle ipotesi non è il gate mancato. È che
**non misuravano ciò che dichiaravano di misurare**. In ordine di gravità:

| Esito | Bracci | Motivo |
|---|---|---|
| **Non eseguito** | P-F | il checkpoint selezionato è R0 all'epoca 1, con peso ausiliario esattamente 0 |
| **Non risolvibile** | R0, D, S0, S1 | le differenze fra bracci sono sotto il rumore dello strumento, e il loro ordinamento non è invariante alla statistica riassuntiva |
| **Meccanismo mai acceso** | D, S1 | gate a CV 2–7%, movimento prevalentemente di modo comune; nessun effetto sulla traiettoria di ottimizzazione |
| **Meccanismo acceso, non trasferito** | S0 | il predictor impara una dinamica causale che generalizza, ma la rappresentazione condivisa non si muove |

Nessun risultato della prima esecuzione viene citato altrove come evidenza a favore o contro le
ipotesi P, D o S. I run restano archiviati con i propri artifact; le cifre riportate qui sotto sono
sufficienti a ricostruire l'argomento senza riaprirli.

### 1.1 Il reperto primario: l'ordinamento dei bracci non è invariante

Macro-F1 validation, seed 42, stessi 2.995 campioni, stesso split, stesso ordine dati.

| Run | epoche | massimo | media ultime 16 | media ep. 10–64 | max − ultime16 |
|---|---:|---:|---:|---:|---:|
| C0 seed 42 | 128 | 55,18 | 54,36 ± 0,49 | — | +0,82 |
| C0 seed 43 | 128 | 54,88 | 53,81 ± 0,51 | — | +1,07 |
| C0 seed 44 | 128 | 56,56 | 55,48 ± 0,52 | — | +1,08 |
| R0 recipe v1 | 64 | 54,97 | 53,75 ± 0,63 | 53,15 | +1,22 |
| P-F | 64 | 54,97 | 54,01 ± 0,52 | 53,21 | +0,96 |
| D recipe v1 | 64 | 55,57 | 54,28 ± 0,36 | 53,55 | +1,29 |
| R0 recipe v2 | 64 | 55,30 | 54,54 ± 0,39 | 54,59 | +0,75 |
| D recipe v2 | 64 | 55,48 | 54,78 ± 0,42 | 54,55 | +0,71 |
| S0 | 64 | 56,02 | 54,72 ± 0,28 | 54,87 | +1,30 |
| S1 | 64 | 55,75 | 54,55 ± 0,35 | 54,55 | +1,21 |

Il confronto D-v2 contro R0-v2 vale **+0,24 pp** sulla finestra delle ultime 16 epoche e **−0,04 pp**
sulla finestra 10–64. Il confronto S0 contro D-v2 cambia di segno fra le due finestre. Nessuna delle
statistiche è scorretta: è la differenza fra i bracci a non esistere a una scala misurabile.

Il `max − ultime16` è sistematicamente +0,71…+1,30 pp su **tutti** i run, incluso C0. Il valore di
riferimento 55,18% di C0 è esso stesso inflazionato di circa 0,8 pp dalla selezione del massimo su
128 valutazioni rumorose.

McNemar esatto appaiato, seed 42:

```
S0  vs R0-v2 : +0,77 pp   p = 0,25    (197 vs 174 disaccordi)
S1  vs R0-v2 : +0,43 pp   p = 0,52
D-v2 vs R0-v2: +0,10 pp   p = 0,92
D-v1 vs R0-v1: +0,53 pp   p = 0,50
P-F  vs R0-v1: +0,00 pp   p = 1,00    (0 disaccordi su 2.995)
```

Il churn fra bracci è 180–260 campioni su 2.995 (6–9%); i delta netti sono 3–34 campioni. La
deviazione standard appaiata è ≈0,64 pp, quella fra seed 0,90 pp, quella per epoca ≈0,5 pp.

---

## 2. Difetti di implementazione

Sono difetti del codice, non scelte scientifiche. Vanno corretti in ogni caso.

### D1 — L'epoca 1 è selezionabile con obiettivo ausiliario spento

`PredictiveTrainingObjective.effective_weight` in `src/etsr/training/predictive.py`:

```python
progress = (epoch - 1) / (self.ramp_epochs - 1)
return self.weight * min(1.0, max(0.0, progress))
```

Con `ramp_epochs: 4` l'epoca 1 ha peso **esattamente 0**. Sotto la ricetta v1, che degrada C0 dalla
seconda epoca, l'epoca 1 è il massimo di validation. Il checkpoint P-F selezionato è quindi il
checkpoint R0 dopo un'epoca senza obiettivo predittivo: identico a R0 fino all'ultima cifra del
Macro-F1, identico su tutti e cinque i punti della curva prefissi, con **zero** disaccordi per
sample. Il comportamento era intenzionale e asserito in `tests/test_predictive_temporal.py`; la
conseguenza sulla selezione non era stata prevista.

### D2 — Il preflight non può rilevare un obiettivo senza autorità

`run_predictive_preflight` in `src/etsr/evaluation/predictive_diagnostic.py` filtra i gradienti sui
soli nomi dei moduli nuovi (`predictive_head`, `content_router`, `predictor_*`, `surprise_router`).
Verifica che il predictor **abbia** gradienti; non verifica mai che l'obiettivo raggiunga i
parametri condivisi, né il suo rapporto con il gradiente di classificazione sullo stesso supporto.

Per un obiettivo senza parametri nuovi — è il caso di `late_prefix` — il dizionario dei gradienti è
vuoto e il gate **passa a vuoto**.

### D3 — Le norme di gradiente registrate non sono confrontabili

`_objective_gradient_diagnostics` in `src/etsr/training/engine.py` calcola
`classification_gradient_norm` e `auxiliary_gradient_norm` su **supporti parametrici diversi**: la
ausiliaria include i ~89k parametri del predictor, la CE include la testa di classificazione. Il
rapporto fra le due norme globali è un limite superiore del rapporto sui parametri condivisi, non
una sua misura, e non deve essere usato per affermare che l'ausiliaria è N volte più debole.

Il `classification_auxiliary_gradient_cosine`, calcolato sui soli parametri che ricevono entrambi i
gradienti, resta valido.

### D4 — Il profiler non addebita il gating di S1

In `src/etsr/profiling/hardware.py` le operazioni `tap_gate_multiply` e `sigmoid` sono contabilizzate
solo dentro `if module.dynamic_routing:`. Con `surprise_routing=True` e `dynamic_routing=False` il
blocco è saltato e il blocco surprise imposta esplicitamente `surprise_gate_multiplies = 0`, mentre
`forward_sequence` applica realmente i gate e valuta il sigmoid.

| | D recipe v2 | S1 |
|---|---:|---:|
| `sigmoid_per_sample` | 51.200 | **0** |
| `elementwise_multiply` | 3.932.160 | **15.360** |

Mancano circa 3,9 M moltiplicazioni e 51 k sigmoid per sample, pari a ≈14,5 µJ sul proxy Horowitz
contro 7.165 µJ di aritmetica coperta: **≈0,2%**. Non cambia nessuna conclusione energetica, ma è
un errore di contabilità in una sezione che vive sulla promessa di non sovradichiarare.

### D5 — Il rapporto fra learning rate discriminativi collassa a fine schedule

`make_scheduler` applica lo stesso `eta_min = 1e-6` a entrambi i gruppi di parametri. Il rapporto
fra `new_parameter_learning_rate` e `inherited_learning_rate` non è 10 per tutte le 64 epoche:

```
ep  4  →  10,00      ep 48  →  7,11      ep 60  →  2,26
ep 20  →   9,81      ep 56  →  4,15      ep 64  →  1,06
```

La separazione esiste per circa i primi due terzi dello schedule e svanisce nell'ultimo terzo. Non
riteniamo questo responsabile dei risultati, ma la ricetta non può essere descritta come
«learning rate discriminativi per tutte le 64 epoche».

---

## 3. Difetti di formulazione dell'obiettivo e dei target

Sono scelte scientifiche che l'evidenza raccolta mostra essere mal poste.

### F1 — Il target di P-F è quasi temporalmente imprevedibile

Dal probe affine `1×1` in
`artifacts/superseded/predictive_diagnostics/fine_future_probe.json`, contesto
student stage1 → target MG fine pre-LIF a t+2:

```
probe lineare                   R² = 0,0696   nRMSE = 0,949   cosine = 0,303
baseline media di training      R² = −0,0001
baseline privilegiata persistenza (target fine a t)   R² = −0,556
```

La terza riga è quella decisiva: il target fine del teacher al tempo t predice il proprio valore a
t+2 **peggio della media globale**. A 100 ms lo stream fine è quasi bianco. Con una loss della
famiglia L2 il predittore ottimale è la media condizionata, cioè approssimativamente una costante,
ed è coerente con quanto osservato: la predictive loss di P-F scende del 10% (0,3215 → 0,2893) e si
appiattisce entro tre epoche.

Il probe è un limite inferiore affine e non falsifica il principio; ma non era sufficiente a
giustificare l'allocazione di un braccio a `h=2` senza prima verificare orizzonti alternativi.

### F2 — Metà dell'obiettivo di S è speso sulla coda banale

`temporal_auxiliary_statistics` media `prediction_active` e `prediction_tail` a peso uguale, e
`_temporal_prediction_region_values` media stage1 e stage2 a peso uguale. La coda è il **44,7%**
della finestra (22,1 bin attivi su 40 in mediana) e vi è quasi interamente prevedibile:

| `tcap_predictive_probe.json` | regione attiva | coda |
|---|---:|---:|
| stage1, nMSE predittore convesso | 0,857 (persistenza 0,956) | **0,104** |
| stage2, nMSE predittore convesso | 0,961 (persistenza 1,257) | 0,590 |

La firma si ritrova nel training di S0: skill contro persistenza **0,437 nella coda** contro
**0,266 nella regione attiva**. Circa metà dell'obiettivo è allocata a estrapolare il decadimento di
un segnale privo di eventi.

### F3 — S1 è stato costruito su una relazione che P0 non aveva validato

`history_damage` in `tcap_predictive_probe.json` riporta, fra surprise e danno da rimozione della
storia, una correlazione di Pearson 0,282 e una **parziale di 0,196** al netto di ampiezza ed event
rate. Il segnale è reale ma la quantità correlata è il danno da rimozione di **tutta** la storia,
aggregata per sample: P0 ha validato `surprise → quanta storia serve`, non
`surprise → quale ritardo scalare`.

S1 è stato costruito sulla seconda relazione. L'esito misurato è quello atteso dalla prima.
Decomposizione dei gate S1 in modo comune e differenziale:

| | modo comune (media−1) | differenziale (sd fra delay) | dipendenza dal contenuto (sd within-sample) | comune/diff |
|---|---:|---:|---:|---:|
| ep. 15, stage2 | +0,1231 | 0,0215 | 0,0096 | 5,7 |
| ep. 64, stage2 | +0,1857 | 0,0137 | 0,0148 | **13,6** |
| ep. 64, stage1 | +0,0392 | 0,0075 | 0,0237 | 5,2 |

Nello stage profondo, a fine training, il router sposta il livello comune 13,6 volte più di quanto
differenzi fra i quattro ritardi: circa il 93% «più memoria» e il 7% «quale memoria».

### F4 — Il residuo di S1 viene mediato nello spazio prima del router

In `src/etsr/models/temporal.py`, `_conditional_gates` calcola
`error = (sequence - prediction).abs().mean(dim=spatial_dims)` e alimenta un `Linear(C → K)`. I gate
risultanti sono **spazialmente uniformi**. È esattamente ciò che la scelta di un router locale per D
intendeva evitare, con la motivazione registrata che il pooling globale diluisce una transizione
confinata alla bocca. Incoerenza interna fra i due bracci.

### F5 — I gate di D confondono quantità e allocazione della memoria

Con gate indipendenti `g_d = 2σ(z_d)` nulla obbliga il router a scegliere fra i ritardi: può
aumentarli tutti insieme, ed è ciò che accade. Il modulo risolve simultaneamente «quanta memoria
usare» e «come distribuirla fra 50, 100, 200 e 400 ms», due gradi di libertà non identificabili
nella stessa variabile.

### F6 — Il routing dinamico nello stage1 è strutturalmente quasi inutile

Dai coefficienti convessi del probe TCAP, numero efficace di tap per canale:

| | exp(entropia) | participation ratio | massa media su d=1 | canali con argmax su d=1 |
|---|---:|---:|---:|---:|
| stage1 | 2,23 | 1,70 | 0,759 | 64 su 64 |
| stage2 | 3,43 | 3,06 | 0,461 | 128 su 128 |

Lo stage1 non è «quasi statico»: è quasi monodelay. Applicarvi un router dinamico aggiunge
parametri e costo senza una scala temporale da selezionare.

### F7 — Il target di S premia la ridondanza temporale

La loss ricostruisce `sg(x_t)` dai propri tap. Il gradiente entra nell'encoder solo attraverso
`x_{t-d}`, perché il target è staccato; la pressione al collasso passa quindi da un solo lato ed è
più debole di quanto una lettura ingenua suggerisca. Resta però che **l'ottimo dell'obiettivo è una
rappresentazione più auto-prevedibile**, mentre per il lip reading l'informazione discriminativa
potrebbe risiedere nell'innovazione `r_t = x_t − x̂_t`. Questa alternativa non è mai stata misurata
ed è oggi la principale incognita della fase.

---

## 4. Difetti di misura e diagnostica

### M1 — La selezione per massimo su epoche rumorose

Documentato in 1.1. Non è un difetto del gate ma della **statistica**: il massimo su 55–128
valutazioni con σ ≈ 0,3–0,6 pp sovrastima il livello del modello di 0,7–1,3 pp, in modo diverso da
braccio a braccio perché dipende dal numero di occasioni e dalla varianza locale.

### M2 — Le epoche non sono repliche statistiche

Checkpoint consecutivi sono fortemente autocorrelati. Una media tardiva è una **statistica di
stabilità**, non un campione di n osservazioni indipendenti, e non se ne può derivare un errore
standard come `σ/√n`. L'unità statistica per un'affermazione di superiorità resta il **seed**.

### M3 — «Stessa cross-entropy» non dimostra «stessa rappresentazione»

Le traiettorie di classification loss dei bracci sono indistinguibili:

```
mean |CE_S0  − CE_R0v2| su 64 epoche = 0,00113
mean |CE_Dv2 − CE_R0v2| su 64 epoche = 0,00284
escursione totale di CE_R0v2          = 0,03517
```

È un'evidenza forte che l'ottimizzazione non è stata deviata, ed è indipendente dal problema di
supporti di D3. Ma la cross-entropy è una proiezione molto compressa della funzione del modello, e i
bracci producono centinaia di predizioni diverse. La formulazione difendibile è: **l'obiettivo
ausiliario perturba il modello senza produrre una deviazione orientata**, coerentemente con il fatto
che i disaccordi sono simmetrici (197 contro 174 fra S0 e R0-v2). Per affermare che il backbone non
si muove servono misure dirette: spostamento relativo dei parametri per layer e CKA/coseno delle
feature.

### M4 — Le sostituzioni su checkpoint co-adattato confondono semantica e scala

La diagnostica a gate costanti sostituisce i gate con la loro media di training. Il suo delta scala
con il learning rate: **+1,53 pp** con la ricetta a `1e-4`, **+0,71 pp** con quella a `1e-5`, sullo
stesso meccanismo. Misura quanto la rete si è co-adattata ai gate, non quanta informazione i gate
trasportano.

Lo stesso confondente invaliderebbe qualsiasi ablazione che sostituisca `x_t` con `x̂_t` o con `r_t`:
le due componenti hanno norme diverse da `x_t` e l'input a valle risulterebbe fuori distribuzione in
scala. Per misurare contenuto informativo si usano **probe**, non sostituzioni.

### M5 — La varianza del target non basta come rilevatore di collasso

`temporal_target_active_variance` sale da 1,3746 a 1,4374 durante S0. Ma una pressione predittiva
maggiore potrebbe mantenere alta la varianza fra campioni e canali rendendo le feature semplicemente
**più lente**. Serve una metrica di variazione temporale normalizzata, non solo di varianza.

### M6 — Statistiche appaiate per sample mai prodotte

Le predizioni per sample esistono in `validation_shortcuts.csv` ma non sono mai state usate per
confronti appaiati, né per la decomposizione Acc1/Acc2. Quando calcolata a posteriori, quella
decomposizione mostra che il guadagno di S0 si concentra sulle parole **non** confondibili:

| | Acc1 (50 parole confondibili) | Acc2 (50 comuni) |
|---|---:|---:|
| R0-v2 | 46,49 | 64,62 |
| S0 | 46,83 (+0,34) | **65,82 (+1,20)** |
| S1 | 46,69 | 65,29 |
| D-v2 | 46,89 | 64,42 |

Acc1 è la metrica che dipende dal timing fine (`taking/taken`, `spend/spent`, `allow/allowed`). È lì
che un meccanismo temporale dovrebbe manifestarsi, ed è lì che non si manifesta.

---

## 5. Difetti di protocollo

### P1 — La soglia decisionale era sotto il rumore del proprio strumento

Il gate di +1 pp su seed singolo va confrontato con una deviazione standard appaiata di 0,64 pp, una
deviazione fra seed di 0,90 pp e una per epoca di ≈0,5 pp. Il protocollo non poteva distinguere un
vero +0,8 pp da zero. Il gate non ha respinto S0: non aveva la risoluzione per giudicarlo.

La correzione non è spostare la soglia. Vedi sezione 9.

### P2 — P-F resta inconcludente, non negativo

Sotto la ricetta v1 la predictive loss è attiva dall'epoca 2 e nessuna epoca successiva batte
l'epoca 1; la media delle ultime 16 epoche di P-F (54,01) supera quella di R0-v1 (53,75) di 0,26 pp.
Ma il substrato subisce contemporaneamente un reheating patologico del learning rate, e il
checkpoint selezionato ha peso ausiliario nullo. **P-F-v1 non è un test dell'ipotesi.** Non è
neppure una prova che l'ipotesi sia falsa.

### P3 — Il budget era esaurito

Circa 47 GPU-ore spese contro 44,86 pianificate e 53,83 di tetto invalicabile, per zero risultati
interpretabili.

**Superato il 2026-09-24:** il vecchio tetto numerico non governa la nuova campagna
(`DECISIONS.md`), ma non è stato sostituito da un budget indefinito. La disciplina è il programma
chiuso della sezione 12.7: seed 42, stop rule meccanicistiche e repliche soltanto per i bracci
positivi.

---

## 6. Incoerenze documentali

| Documento | Incoerenza |
|---|---|
| `ROADMAP.md` riga 34 | «8 continuazioni da 32 epoche» contro le 64 del protocollo attivo |
| `ROADMAP.md` riga 37 | «tre volte il training del C0, circa 26,92 ore» contro 5×/6× e 44,86/53,83 ore |
| `DECISIONS.md` punto 5 | «otto continuazioni da 32 epoche» |
| `PROJECT_STATUS.md` | dichiara che nessun training è stato lanciato; sette lo sono stati |
| `EXPERIMENT_LEDGER.md` | non contiene alcuna voce della fase predittiva |

Riallineati nel commit `ca8c0ea`; `TRAINING_RECIPE.md` è stato riallineato il 2026-09-24.

---

## 7. Contratto di correzione

### 7.1 Correzioni obbligatorie

Non negoziabili, indipendenti da qualunque scelta scientifica.

| ID | Correzione | Dove |
|---|---|---|
| C1 | Escludere dalla selezione del candidato ogni epoca in cui il peso ausiliario è inferiore al peso nominale. L'epoca resta valutata e registrata come warm-up, non come candidato. | `runner.py`, selezione del best |
| C2 | Estendere il preflight con l'audit di gradiente sui **parametri condivisi**: per blocco, rapporto `‖λ∇L_aux‖ / ‖∇L_CE‖` e coseno, con ausiliaria separata fra regione attiva e coda. Il gate fallisce se l'obiettivo non produce gradiente misurabile sui parametri condivisi. | `evaluation/predictive_diagnostic.py` |
| C3 | Il preflight fallisce se una configurazione dichiara un obiettivo ausiliario e nessuna famiglia di gradienti viene misurata. Chiude il passaggio a vuoto di `late_prefix`. | idem |
| C4 | Sostituire le norme globali di `_objective_gradient_diagnostics` con rapporti per blocco sullo **stesso supporto**; conservare il coseno condiviso. | `training/engine.py` |
| C5 | Spostare `tap_gate_multiply` e `sigmoid` fuori dal blocco `dynamic_routing`, così che qualunque percorso di gating venga addebitato. | `profiling/hardware.py` |
| C6 | Rendere `eta_min` proporzionale al learning rate di ciascun gruppo, così che il rapporto discriminativo sia costante per l'intero schedule. | `training/engine.py` |
| C7 | Registrare per epoca: variazione temporale normalizzata `V_Δ = E[‖x_t − x_{t-1}‖² / (‖x_t‖² + ε)]` nella regione attiva, per stage; rapporti e coseni di C4; skill attiva e di coda separate. | `runner.py`, `models/temporal.py` |
| C8 | Registrare per ogni run, oltre al best: media e deviazione della finestra tardiva su una finestra dichiarata prima del lancio, e le predizioni per sample già disponibili. | `runner.py`, `summary.json` |

### 7.2 Revisioni di progetto

Cambiano **che cosa** l'esperimento verifica. Sono giustificate dall'evidenza di questo documento e
vanno dichiarate come tali: il braccio riesegue sotto lo stesso nome un'ipotesi più precisa.

| ID | Revisione | Motivo |
|---|---|---|
| R1 | Pesi espliciti e configurabili per regione (attiva/coda) e per stage nell'obiettivo S; default a dominanza della regione attiva. Le due regioni restano registrate separatamente. | F2 |
| R2 | Routing dinamico limitato allo **stage2**. Lo stage1 conserva TCAP fisso. | F6 |
| R3 | Separare ampiezza e allocazione: `y_t = x_t + a_t Σ_d K π_{t,d} W_d x_{t-d}` con `a_t ≥ 0`, `π = softmax_d`, `K = |delays|`, inizializzati a `a=1` e `π` uniforme così che la funzione iniziale resti C0. | F3, F5 |
| R4 | Il residuo che alimenta il router conserva la struttura spaziale, coerentemente con la scelta locale già adottata per D. | F4 |
| R5 | P-F non viene rieseguito all'orizzonte `h=2` prima di una nuova verifica di fattibilità del target su orizzonti alternativi e sul futuro coarse. | F1, P2 |

### 7.3 Revisioni dopo l'audit (2026-09-24)

Motivate dagli esiti della sezione 12. Aggiornano R1 e risolvono R5.

| ID | Revisione | Motivo |
|---|---|---|
| R1′ | Nell'obiettivo S la coda ha peso **0**, non 0,25. | F2: la coda è la parte banale da predire (nMSE 0,10 nello stage1); A2: nella coda x̂ porta più informazione di classe di x (8,89% contro 6,64%). A1 stratificato non la sostiene: nello stage2 la coda è neutra (+0,02) |
| R6 | Il predittore causale esiste **solo nello stage2** (`temporal_channel_mixer_predictive_stages: [2]`); `V_Δ` resta registrata anche nello stage1. | A2: informazione di classe 24% nello stage2 contro 5% nello stage1, dove il residuo (il movimento) è il portatore migliore |
| R7 | Autorità esplicita: λ calibrato prima dell'epoca 1 perché il rapporto `‖λ∇L_aux‖/‖∇L_CE‖` sul backbone condiviso valga 0,25, e rimisurato ogni epoca su un batch fisso stratificato, con passo massimo ×2. Il preflight esige un rapporto minimo di 0,05 per ogni obiettivo ausiliario dichiarato. | A1, A3 |
| R8 | D, S0 e S1 si addestrano **da zero** con la ricetta congelata di C0; i controlli sono i seed archiviati di C0. L15 resta una continuazione con controllo R0. | A3: il substrato di continuazione non si muove (CKA 0,999), e lavora a 1/30 del learning rate con cui la rappresentazione si è formata |
| R9 | L15: distillazione del solo prefisso a 1,5 s, letto con denominatore fisso di 40 bin. | durate: 30,8% delle parole finite a 1 s, 99,4% a 1,5 s; A4: l'88% della variazione del logit vero fra 1,5 e 2 s è diluizione del readout |
| R5 risolta | Famiglia di target fine **chiusa**; target coarse **sospeso** a h=2. | probe R5, sezione 12.5 |

---

## 8. Audit obbligatorio prima di qualunque training

Nessuna GPU-ora di training viene spesa prima che questo audit, interamente **checkpoint-only** su
C0, R0-v2 e il best S0 archiviati, abbia prodotto le sue quattro sezioni. Costa inferenza e backward
su qualche centinaio di campioni del development-train, con holdout disgiunto per sample.

**A1 — Autorità del gradiente.** Per blocco (`stage1` condiviso, `W_d` di TCAP1, `stage2` condiviso,
`W_d` di TCAP2, testa) e separando l'ausiliaria fra regione attiva e coda:

```
ρ_l   = ‖λ ∇_{θ_l} L_aux‖ / ‖∇_{θ_l} L_CE‖
cos_l = cos(∇_{θ_l} L_CE, ∇_{θ_l} L_aux)
```

Risponde a: l'obiettivo raggiunge l'encoder, e dove.

**A2 — Dove vive l'informazione discriminativa.** Sul best S0 congelato, decomponendo
`x_t = x̂_t + r_t` a ciascun mixer, quattro probe lineari con lo stesso readout del classificatore,
fittati sul development-train con holdout disgiunto per sample:

```
Acc(x)      Acc(x̂)      Acc(r)      Acc([x̂, r])
```

separatamente per stage1/stage2 e regione attiva/coda. Il quarto probe misura la complementarità
delle due componenti. **Probe, non ablazioni**: vedi M4.

**A3 — Movimento della rappresentazione.** Fra S0 e R0-v2: spostamento relativo dei parametri per
layer `‖θ^{S0}_l − θ^{R0}_l‖ / ‖θ^{R0}_l‖`, e CKA/coseno delle feature sugli stessi campioni.
Risponde a M3 senza inferire dalla cross-entropy.

**A4 — Da dove vengono i +7,80 pp fra 1,5 s e 2,0 s.** Il 30,8% dei campioni è terminato entro 1 s,
il **99,4% entro 1,5 s**. Il guadagno da 47,50% a 55,30% di Macro-F1 nell'ultimo mezzo secondo non
può derivare da eventi nuovi. Poiché il readout è una media su finestra fissa seguita da uno strato
lineare, i logit si decompongono esattamente:

```
z_{1:40} = (30/40) z_{1:30} + (10/40) z_{31:40}
```

Si misura quindi il contributo dei bin 31–40 al **margine di classe**
`Δm_tail = m_{1:40} − m_{1:30}`, separando logit della classe vera, miglior competitore, margine e
decomposizione Acc1/Acc2. Se la coda aumenta sistematicamente il margine, quei 500 ms contengono
settling discriminativo; se il contributo medio è nullo, il guadagno è riduzione di varianza
dell'estimatore e l'eventuale supervisione dei prefissi perderebbe la propria motivazione.

### 8.1 Come l'audit seleziona il prossimo training

L'esito non è una regola automatica, ma restringe lo spazio in modo netto:

- `ρ` piccolo sui parametri condivisi **e** `Acc(x̂) ≫ Acc(r)` → l'obiettivo punta alla componente
  giusta senza autorità: ha senso un budget di gradiente esplicito.
- `Acc(r) ≳ Acc(x̂)`, soprattutto nello stage2 attivo → **non si aumenta il peso**: minimizzare il
  residuo sarebbe attivamente controproducente, e la revisione riguarda l'esposizione di `r_t` al
  classificatore.
- Informazione predittiva utile concentrata nello stage2 → qualsiasi nuova loss predittiva diventa
  stage2-only, coerentemente con R2.
- A4 mostra settling discriminativo reale nella coda → la supervisione tardiva a 1,5 s diventa il
  candidato prioritario.

---

## 9. Come si decide adesso

**La soglia preregistrata di +1 pp su seed singolo viene abbandonata come regola di passaggio.** Non
viene sostituita da un'altra soglia. L'evidenza di 1.1 mostra che nessuna soglia applicata a una
statistica di un singolo run ha la risoluzione necessaria, e che l'ordinamento dei bracci non è
neppure invariante alla scelta della statistica.

La disciplina si sposta dalla soglia al **corredo di evidenza obbligatorio**. Per ogni braccio, prima
di qualunque decisione, devono essere prodotti e letti insieme:

1. massimo, ultima epoca, media e deviazione della finestra tardiva dichiarata prima del lancio, e
   il valore corrispondente del controllo appaiato;
2. confronto appaiato per sample contro il controllo, con conteggi di disaccordo;
3. decomposizione Acc1/Acc2;
4. curva ai prefissi e, quando pertinente, decomposizione del margine di coda;
5. le diagnostiche di meccanismo del braccio: per un obiettivo ausiliario `ρ` e coseno per blocco,
   `V_Δ`, skill attiva e di coda; per un router, modo comune contro differenziale e dipendenza dal
   contenuto;
6. costo in GPU-ore e profilo del proprio checkpoint.

La decisione è presa **caso per caso**, argomentata e registrata in `DECISIONS.md` con il
ragionamento, non con il solo esito numerico. Una differenza prestazionale che non sia accompagnata
da una firma meccanicistica coerente con ciò che il braccio dichiara di fare non viene attribuita al
meccanismo. Una firma meccanicistica forte senza differenza prestazionale viene registrata come
risultato meccanicistico e non come candidato.

Restano invece fermi due vincoli:

- l'official test resta escluso;
- un'affermazione di superiorità richiede la conferma su più seed, perché l'unità statistica è il
  seed e non l'epoca (M2).

---

## 10. Politica di sovrascrittura e riesecuzione

La prima esecuzione viene **sovrascritta**, non affiancata.

**Configurazioni.** I file `configs/dvslip_predictive_*.yaml` vengono modificati in luogo. Non si
creano nomi nuovi per le versioni corrette. I `recipe_id` e i nomi di esperimento restano quelli
originali (`dvslip_predictive_r0`, `dvslip_predictive_s0`, `dvslip_predictive_s1`,
`dvslip_predictive_dynamic_tcap`, `dvslip_predictive_fine_future`, `dvslip_predictive_late_prefix`).

**Artifact.** I nuovi run usano gli stessi nomi di esperimento, senza suffissi aggiuntivi. Il suffisso
temporale è generato automaticamente e distingue le esecuzioni.

**Artifact della prima esecuzione.** Vengono spostati sotto `artifacts/superseded/` anziché
eliminati. Le cifre di questo documento derivano da essi e non sarebbero più verificabili
altrimenti; inoltre `PREDICTIVE_TEMPORAL_ROADMAP.md` §13 vincola a non cancellare esperimenti validi
ma negativi. Restano archiviati come evidenza **di questo audit**, non come evidenza sulle ipotesi
scientifiche.

**Conseguenza dichiarata.** Poiché le revisioni della sezione 7.2 cambiano ciò che alcuni bracci
verificano, un braccio rieseguito sotto il nome originale **non è confrontabile** con l'omonimo della
prima esecuzione. I risultati della prima esecuzione non vengono citati come baseline, come
riferimento o come tendenza. Questo documento è il solo luogo in cui quelle cifre compaiono.

**Ordine.** L'audit della sezione 8 precede qualunque training. La riesecuzione non riparte da sette
continuazioni: segue il programma chiuso della sezione 12.7, deciso sull'esito dell'audit. Il valore
numerico di budget di §5 P3 non viene riutilizzato come soglia della nuova campagna.

**Regime.** Per R8 alcuni bracci non sono più continuazioni: D, S0 e S1 si addestrano da zero con la
ricetta di C0. Conservano i nomi di esperimento originali; la regola di non confrontabilità con la
prima esecuzione vale a maggior ragione.

---

## 11. Che cosa resta invariato

Non tutto della prima esecuzione è da rifare. Restano validi e non vanno riaperti:

- il contratto di causalità: `stage1_causal_prefix_max_abs_difference = 0,0` e
  `encoded_causal_prefix_max_abs_difference = 0,0` in tutti i preflight; teacher in `eval()` con
  `requires_grad_(False)`; nessuna dipendenza dal futuro nei percorsi dichiarati causali;
- il warm-start esatto: `initial_logits_max_abs_difference = 0,0`, funzione iniziale identica a C0 in
  tutti i bracci;
- la fedeltà architetturale alla specifica: 17.152 parametri per il predictive head, 10.728 per i
  due router di contenuto, 88.832 per il predictor MIMO, 768 per la proiezione del residuo,
  esattamente come preregistrato;
- la struttura degli artifact, il manifest, gli hash dei checkpoint e la separazione fra parametri
  deployati e training-only;
- il riferimento C0 e il suo protocollo dati, con l'avvertenza di 1.1 sul fatto che 55,18% è un
  massimo e che il livello tardivo del medesimo run è 54,36 ± 0,49.

Va inoltre conservata la distinzione fra i due esiti meccanicistici, perché nella tesi vanno
raccontati diversamente:

- **D** ha imparato un routing dipendente dall'input che la rete utilizza, ma di ampiezza troppo
  piccola e con utilità downstream non misurabile;
- **S1** ha un segnale di discrepanza reale, ma la mappa di compressione e routing scelta lo
  collassa in un'amplificazione globale della memoria;
- **S0** ha imparato una dinamica causale che generalizza — skill 0,266 contro persistenza nella
  regione attiva, train e validation coincidenti — senza che questa competenza si trasferisca al
  classificatore.

Sono tre risultati diversi e nessuno dei tre è «l'idea non funziona».

---

## 12. Esito dell'audit (2026-09-24)

Report: `artifacts/predictive_phase1_audit/phase1_audit.json`, commit `ca8c0ea`, worktree pulito,
sha256 di C0 verificato, nessun accesso all'official test. A1, A3 e A4 coincidono fino all'ultima
cifra fra due esecuzioni indipendenti. A2 è stato rieseguito con 8192 campioni di fit e 2048 di
holdout: con 512/256 violava l'ordinamento `Acc([x̂, r]) ≥ Acc(x)` imposto dalla sua stessa classe
di ipotesi, ed era quindi dominato dalla varianza.

### 12.1 A1 — Autorità e direzione del gradiente ausiliario della S0 archiviata

Misurato su quattro batch disgiunti stratificati per classe, 64 parole distinte, commit `36525a8`:

| blocco | ratio active | coseno active | ratio tail | coseno tail |
|---|---:|---:|---:|---:|
| stage1_shared | 8,40e-05 | −0,532 | 8,44e-05 | +0,276 |
| tcap1_weights | 9,55e-05 | −0,156 | 9,46e-05 | −0,003 |
| stage2_shared | 4,73e-05 | −0,017 | 3,82e-05 | +0,020 |
| tcap2_weights, head | 0 | — | 0 | — |
| aggregato condiviso | 8,07e-05 | −0,410 | 8,01e-05 | +0,201 |

Il segno è lo stesso in tutti e quattro i batch: nella regione attiva da −0,23 a −0,51, nella coda
da +0,09 a +0,33.

- **Magnitudine confermata**: 4–10 × 10⁻⁵, anche indipendentemente da A3.
- **La prima versione di A1 aveva il segno opposto** (attiva +0,425, coda −0,263) perché misurava
  16 campioni della sola parola «accused»: DVS-Lip è ordinato per classe e il batch diagnostico era
  il primo non mescolato. Quella struttura era un artefatto ed è ritirata, insieme a ogni
  conclusione che ne derivava.
- La componente attiva dell'obiettivo archiviato si oppone al gradiente di classificazione quasi
  solo attraverso lo stage1 (−0,53); sullo stage2 è neutra (−0,02). È coerente con A2, per cui nello
  stage1 l'informazione di classe sta nel residuo, e rafforza R6, che rimuove il predittore di
  stage1.
- **Limite dell'interpretazione.** Il riferimento è il gradiente CE di *training* su un checkpoint
  al 92–93% di accuracy di training e 55% di validation. Opporsi a quella direzione può voler dire
  danneggiare la classificazione oppure contrastare la memorizzazione: il coseno non distingue i due
  casi. La quantità che li distingue è l'allineamento con il gradiente della loss su campioni non
  usati per il training, che è anche la derivata al primo ordine della loss di validation rispetto
  al peso ausiliario.

### 12.2 A2 — Dove sta l'informazione discriminativa

Accuracy di sonde lineari su holdout (100 classi, caso 1%, deviazione binomiale 0,5–1,0 pp):

| stage | regione | x | x̂ | r | [x̂, r] |
|---|---|---:|---:|---:|---:|
| stage1 | attiva | 5,27 | 5,32 | **6,88** | 7,57 |
| stage1 | coda | 1,76 | 2,20 | 2,00 | 2,69 |
| stage2 | attiva | **24,22** | 23,54 | 22,41 | 25,00 |
| stage2 | coda | 6,64 | **8,89** | 6,40 | 10,94 |

- L'informazione di classe accessibile linearmente sta nello **stage2** (24% contro 5%).
- Nello stage2 attivo x̂ e r portano quasi la stessa informazione, e separarli aggiunge 0,78 pp:
  l'informazione esclusiva del residuo è piccola, quindi il rischio F7 è limitato lì.
- Nello stage1 il residuo è il portatore migliore: con un predittore quasi mono-ritardo, r è la
  derivata temporale, cioè il movimento. Una loss predittiva sullo stage1 agirebbe da penalità di
  levigatezza proprio su quell'informazione.
- Nella coda la predizione dal passato è più discriminativa del presente che decade.

### 12.3 A3 — Movimento della rappresentazione

CKA lineare fra S0 e R0: 0,999988 e 0,999956 nello stage1, 0,994914 e 0,994897 nello stage2.
Spostamento relativo massimo dei parametri: 1,01%, sulle matrici del mixer di stage2. La prima
esecuzione di S0 non ha modificato la rappresentazione; l'impronta dell'obiettivo è nel punto
giusto ma dell'ordine dell'1%.

### 12.4 A4 — La coda contiene settling discriminativo

| C0, validation | 1,5 s | 2,0 s | Δ | Δ margine medio |
|---|---:|---:|---:|---:|
| accuracy | 48,81% | 55,53% | +6,71 pp | +0,311 (positivo nel 60%) |
| Acc1, parole confondibili | 37,47% | 45,89% | **+8,42 pp** | **+0,478** |
| Acc2, parole comuni | 60,15% | 65,15% | +5,01 pp | +0,145 |

Il meccanismo è la **soppressione dei competitori**: il logit vero scende da 2,398 a 1,720 (×0,717),
il miglior competitore da 2,516 a 1,526 (×0,606); il logit vero sale solo nel 26% dei campioni.
Poiché `z₄₀ = 0,75·z₃₀ + 0,25·z_coda`, l'88% della variazione del logit vero è pura diluizione del
readout: senza correzione un target KL a 2 s mescolerebbe settling e ricalibrazione di scala. R0 è
indistinguibile da C0 su ogni statistica A4.

### 12.5 Probe R5 — Fattibilità dei target

| h | R² coarse | R² fine | coarse / fine |
|---:|---:|---:|---:|
| 1 | 0,323 | 0,096 | 3,35× |
| 2 | 0,249 | 0,070 | 3,58× |
| 4 | 0,182 | 0,054 | 3,36× |
| 8 | 0,125 | — | — |

- Il target fine non supera R² 0,10 a nessun orizzonte (nRMSE 0,935–0,956). **Chiuso.**
- Il vantaggio del coarse è costante fra orizzonti: appartiene alla famiglia di target. A h=8 il
  target coarse ha autocorrelazione 0,08, eppure la sonda — che contiene l'identità nella propria
  classe di ipotesi, perché il teacher coarse è C0 — ne spiega il 12,5%: dinamica lineare, non
  persistenza. **Sospeso** a h=2, per i motivi registrati nel suo `blocked_reason`.
- Il coseno non è confrontabile fra target: sul coarse una costante ottiene già 0,75–0,78.

### 12.6 Verdetti per proposta

| Proposta | Verdetto | Motivo |
|---|---|---|
| L — supervisione tardiva | **da eseguire** come L15 | A4: settling ampio, concentrato su Acc1, meccanismo coerente con la KL |
| S0 — predizione ausiliaria | **da eseguire** da zero con R1′, R6, R7 | A1–A3 spiegano il fallimento; rischio F7 limitato nello stage2 |
| S1 — routing da sorpresa | **da eseguire** da zero, confrontato con S0 | il predittore si addestra anche senza autorità; il routing non era verificabile in continuazione |
| D — TCAP dinamico | **da eseguire** da zero | come S1: in continuazione il router parte da un punto stazionario; C0 è la sua ablazione riaddestrata |
| P-F, P-0 | **chiusi** | target non predicibile |
| P-C | **sospeso** | fattibile; autorità e direzione mai misurate; agisce sullo stage1 |
| configurazioni di fusione | **rimosse** | la sola fusione prevista è strutturata e condizionata a due componenti positivi |

La chiusura di D e S1 proposta inizialmente era basata su evidenza di continuazione, che A3 ha
mostrato non informativa per meccanismi di questo tipo: è stata ritirata.

### 12.7 Programma di riesecuzione

| Braccio | Regime | Controllo | Domanda |
|---|---|---|---|
| L15 | continuazione da C0 | R0 | il settling si può anticipare? |
| D | da zero, ricetta C0 | C0 seed 42/43/44 | il contenuto decide utilmente quanta memoria e quale? |
| S0 | da zero, ricetta C0 | C0 seed 42/43/44 | la predizione modella la rappresentazione se agisce mentre si forma? |
| S1 | da zero, ricetta C0 | S0 e C0 | l'errore predittivo aggiunge valore al controllo della memoria? |

Seed 42 per primo; 43 e 44 per ciò che mostra la firma meccanicistica attesa. S1 viene avviato solo
se S0 dimostra sul validation set abilità predittiva finita e positiva rispetto ai riferimenti
causali: senza un predittore utile, la sorpresa di S1 non ha un meccanismo interpretabile da
instradare. Controlli di attribuzione solo se un braccio funziona: D con gate statici riaddestrato,
per separare adattività e riscalamento; L con distillazione a finestra piena
(`prefix_steps: [40]`, nessun codice aggiuntivo), per separare la supervisione del prefisso dalla
distillazione generica. Dopo due componenti positivi, un'unica fusione strutturata: sorpresa →
ampiezza, contenuto → allocazione.

I bracci da zero partono dall'inizializzazione del backbone della topologia C0 allo stesso seed e dal
suo stesso flusso di dati e augmentation; lo scheduler corrente riproduce esattamente quello con cui
C0 è stato addestrato (test `test_scheduler_reproduces_the_legacy_schedule_of_frozen_c0`). S0 e S1
condividono anche l'inizializzazione del predittore, quindi il loro confronto è appaiato fin
dall'inizio.

### 12.8 Preflight del codice corretto

Eseguiti sul server il 2026-09-24 per i cinque bracci del seed 42, tutti superati; batch diagnostico
di 16 campioni da 16 classi.

| | R0 | L15 | D | S0 | S1 |
|---|---|---|---|---|---|
| inizializzazione | logit = C0 (0,0) | logit = C0 (0,0) | backbone C0 | backbone C0 | backbone C0 |
| causalità architetturale | sì | sì | sì | sì | sì |
| rapporto unitario | — | 1,484 | — | 0,0616 | 0,0616 |
| peso → rapporto condiviso | — | 0,1 fisso → **0,148** | — | 4,06 calibrato → 0,250 | 4,06 → 0,250 |
| coseno condiviso con CE di training | — | **−0,641** | — | −0,037 (inizializzazione casuale) | −0,037 |
| gradiente dei router: inizio → dopo 2 passi | — | — | 0 → 0,42 | — | 0 → 0,55 |

- **L15 ha autorità reale senza calibrazione**: allo stesso peso 0,1 e sullo stesso C0, la KL sul
  prefisso ha un rapporto circa 1.800 volte quello della S0 archiviata (0,148 contro 8,1e-05).
- Il suo gradiente si oppone al CE di training (−0,64, −0,74 sullo stage1). Vale lo stesso limite
  di A1: in regime di memorizzazione è compatibile sia con un danno sia con la regolarizzazione
  tipica della distillazione a temperatura 2. Lo discriminano il run stesso (CE di training e F1
  tardiva contro R0, A4 sul checkpoint) e l'allineamento con il gradiente su dati non visti.
- S0 e S1 hanno calibrazione e inizializzazione identiche: il loro confronto è appaiato.

Il preflight misura anche una condizione che la prima esecuzione non poteva vedere perché congelava
la BatchNorm: **in training la BN non è causale**. `_time_distributed` fonde tempo e batch, e le
statistiche per canale includono i passi futuri; perturbare la seconda metà della sequenza sposta il
prefisso fino a 4,0. Il modello deployato usa statistiche running ed è causale, e C0 è stato
addestrato nello stesso modo: la condizione è comune a controllo e bracci, ed è registrata senza
farne un gate.

### 12.9 Prima ondata corretta, seed 42 (2026-09-25/26)

**L15 completato.** Letto contro R0-v2 archiviato, con cui è appaiato esattamente all'epoca 1: a
1,5 s F1 51,03 contro 47,50 (+3,53); il gap di settling fra 1,5 e 2 s scende da 7,80 a 4,95 pp;
a 2 s la finestra tardiva vale +0,16 pp (McNemar p = 0,35); Acc1 +1,40, Acc2 −0,13. È la firma
attesa (settling anticipato senza costo sul punto finale). Restano da leggere l'R0 di quest'ondata e
A4 sul checkpoint.

**D non è evidenza sull'ipotesi: misura un difetto di parametrizzazione.** Con `amplitude_allocation`
l'ampiezza era `a = exp(clamp(ℓ, −8, 8))`, fino a circa 3.000. Addestrando da zero, l'ampiezza e le
matrici `W_d` sono intercambiabili in scala, e `W_d` parte da zero: nulla trattiene l'ampiezza.
Già nel gate all'epoca 100 l'ampiezza valeva in media 17 (SD 59), ma il gate misura solo la
capacità di adattarsi e lo ha superato. Nel run completo i contributi dei tap valgono 40–150 contro
0,4–1,1 di C0. Esito: best 49,72 (epoca 114), finestra tardiva 48,87 ± 0,50 contro 54,36 ± 0,49 di
C0, accuracy di training 64,1% contro 89,1%.

Correzione: `amplitude_allocation` è ora definita univocamente con `a = 2σ(ℓ) ∈ (0, 2)`.
La memoria totale `Σ_d g_d = aK` resta in `(0, 2K)` come con i gate indipendenti, e
all'inizializzazione `a = 1` riproduce esattamente TCAP fisso. La precedente legge esponenziale è
rimossa: era una parametrizzazione difettosa, non una variante da mantenere. S1 non era ancora
partito. Il run difettoso di D va in `artifacts/superseded/`, e D si riesegue con il proprio
workflow, gate incluso.

**S0: gate fallito solo sulla loss, full run autorizzato.** Il gate (64 campioni, fp32, 500 epoche)
termina con accuracy 95,3% in training e 96,9% in eval, tutto finito, ma CE in eval 1,567 contro la
soglia 1,5; C0 la supera all'epoca 310 (minimo 1,394). S0 raggiunge 0,95 di accuracy all'epoca 290
contro 225. La soglia è stata tarata su training con sola CE. Con un termine ausiliario ad autorità
0,25 sul backbone, una CE meno confidente è l'effetto atteso, non un difetto di cablaggio.
Il gate è deterministico: rilanciato, fallirebbe allo stesso modo. Il full run parte quindi con
`train` e il profiling con `profile-checkpoint` (comandi in
[`OPERATIONS_SMILIES.md`](OPERATIONS_SMILIES.md)). I controlli che il workflow esegue prima del
gate, cioè deriva da C0, report canonico dell'audit e autorizzazione, sono stati rieseguiti sulle
configurazioni correnti, e il preflight è già superato.

Cosa mostra il gate sul meccanismo:

- Il predittore acquista un'abilità vera, non una scorciatoia. A inizializzazione coincide con la
  media dei ritardi (skill 0). Alla fine batte del 26% il riferimento causale migliore, la
  persistenza, che nel frattempo è migliorata perché le feature si sono fatte più lisce (`V_Δ`
  stage2 da 1,50 a 0,44).
- La varianza del target di stage2 passa da 1,67 a 1,30: nessun collasso di scala.
- λ sale da 3,0 a circa 73 (massimo 105) perché il rapporto unitario scende da 0,082 a 0,0034;
  l'autorità effettiva resta 0,25.
- La norma del gradiente globale è simile a quella del gate di C0: il predittore non domina il
  clipping.

Percorsi che il gate non ha esercitato, verificati prima del lancio:

- **AMP.** Il gate girava in fp32, il full run gira in AMP come C0. Sotto autocast la smooth-L1 del
  predittore e la CE sono calcolate in fp32, e la calibrazione disabilita esplicitamente l'autocast.
  Gli overflow fp16 del backward sono gestiti dal GradScaler come per C0 e registrati in
  `amp_overflow_fraction`. Le colonne `train_gradient_*` del batch 0 sono misurate senza scaling in
  fp16, quindi sono approssimate; λ dipende solo dalla calibrazione in fp32.
- **Memoria.** Il picco del gate in fp32 è 12,08 GB (C0: 6,54) su una A4000 da 16,7 GB, e include
  il grafo della calibrazione trattenuto dalla cache del mixer. In AMP il training scende sotto
  quel valore.
- **Valutazione finale.** Curve dei prefissi, prefissi per campione e statistiche predittive sono
  state provate in CPU su un modello con predittore. L'export deployabile è stato esercitato dal
  gate e dal test d'integrazione. La valutazione completa è quella già percorsa da C0, L15 e D.

**128 epoche.** Sono la scelta corretta, per tre ragioni:

1. Il confronto è appaiato con C0: stessa inizializzazione, stesso flusso di dati, stesso schedule
   coseno. Allungare solo S0 confonderebbe durata e meccanismo, e un confronto a durata diversa
   richiederebbe un nuovo controllo C0 alla stessa durata.
2. A 128 epoche C0 è ricotto: learning rate 1e-6, accuracy di training 88,8% → 89,1% nelle ultime
   16 epoche, F1 in plateau da circa l'epoca 96. Il limite è la generalizzazione (89% contro 55%),
   non l'ottimizzazione.
3. Il rallentamento misurato dal gate riguarda la memorizzazione di 64 campioni, e il run stesso
   dice se pesa. Il segnale è una `train_classification_loss` di S0 ancora chiaramente sopra quella
   di C0 all'epoca 128, con l'F1 in salita nell'ultima finestra. In quel caso il seguito è una
   coppia S0/C0 a durata maggiore, non S0 da solo.

Lettura del run S0:

- **Epoca 1.** Il ramp ha peso effettivo 0, quindi l'epoca è attesa identica a C0: train loss
  4,6636, accuracy 0,0108, val loss 4,6544, norma del gradiente 1,4102. Una differenza indica che
  l'appaiamento si è rotto.
- **Dall'epoca 2.** Confrontare `train_classification_loss` (non `train_loss`, che include λ·aux) e
  F1 con C0: epoca 16 3,514 / 14,40, epoca 32 2,387 / 39,89, epoca 64 1,672 / 47,41, epoca 96
  1,398 / 53,83, epoca 128 1,342 / 54,47.
- **Firma meccanicistica.** Skill di validation rispetto al riferimento causale migliore, varianza
  del target (collasso), `V_Δ`, λ e autorità effettiva, `amp_overflow_fraction`.
