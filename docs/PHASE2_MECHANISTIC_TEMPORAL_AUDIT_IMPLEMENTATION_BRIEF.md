# Temporal-Events-Spiking  
## Brief operativo per l’implementazione della Fase 2 — Mechanistic Temporal Representation Audit

> **Destinatario:** agente AI incaricato di modificare il repository `Temporal-Events-Spiking`  
> **Repository di riferimento:** `Bernuz2003/Temporal-Events-Spiking`  
> **Baseline corrente:** Mini-QKFormer-like su DVS-Gesture-Chain order-2  
> **Scopo del documento:** riassumere il lavoro svolto, formalizzare i risultati della Fase 1 e definire in modo implementabile la Fase 2 di audit meccanicistico della rappresentazione temporale.

---

# 1. Contesto scientifico e obiettivo del progetto

Il progetto studia quanto e come le attuali Spiking Transformer frame-based sfruttino la struttura temporale dei dati provenienti da Dynamic Vision Sensors.

La critica di partenza non è che tali modelli ignorino completamente il tempo. I neuroni LIF processano sequenze di timestep e conservano uno stato dinamico. La domanda scientifica è più precisa:

> **Quale livello di dinamica temporale viene realmente rappresentato, quanto tale rappresentazione è robusta, dove viene acquisita o persa nella rete e quanto costo spaziale/computazionale è necessario per ottenerla?**

L’obiettivo finale non è ottimizzare un modello specificamente per DVS-Gesture-Chain. DVS-Gesture-Chain viene utilizzato come ambiente diagnostico controllato perché permette di separare:

- riconoscimento delle primitive presenti;
- riconoscimento del loro ordine;
- robustezza rispetto a trasformazioni temporali;
- conservazione dell’informazione passata;
- integrazione di nuova evidenza.

Le proprietà e le metriche sviluppate devono essere formulate in modo riutilizzabile su altri dataset event-based e sequenziali.

---

# 2. Quanto è stato realizzato finora

Il repository implementa una pipeline sperimentale riproducibile per:

1. caricare DVS-Gesture-Chain;
2. addestrare una baseline Mini-QKFormer-like;
3. separare training, validation e test;
4. selezionare il checkpoint tramite Macro-F1 di validation;
5. calcolare Accuracy, Macro-F1 e confusion matrix;
6. profilare parametri, firing rate, MAC, AC/SOP e una stima energetica Horowitz-style;
7. eseguire perturbazioni temporali a inferenza;
8. salvare artifact JSON, CSV, NPZ e figure;
9. confrontare predizioni originali e perturbate campione-per-campione;
10. valutare la classificazione su prefissi crescenti della sequenza.

La baseline è intenzionalmente una riscrittura modulare e autocontenuta ispirata a Mini-QKFormer. Non è dichiarata come replica bit-exact del repository ufficiale e non deve essere assunta come architettura finale della tesi.

La Fase 1 ha usato:

- DVS-Gesture-Chain;
- catene di due primitive;
- primitive identificate dagli ID `1`, `3`, `8`;
- ripetizioni consecutive disabilitate;
- 6 classi ordinate: `13`, `18`, `31`, `38`, `81`, `83`;
- 16 frame per sequenza;
- seed del modello `42`;
- 120 epoche;
- AdamW;
- selezione del checkpoint tramite Macro-F1 di validation;
- Mini-QKFormer con `embed_dim=128`, 8 teste e 488.582 parametri.

Le semantiche testuali delle primitive non devono essere hard-coded nel nuovo codice. Quando servono, devono essere lette dal mapping originale del dataset.

---

# 3. Risultati principali della Fase 1

## 3.1 Prestazione della baseline

Il miglior checkpoint è stato selezionato all’epoca 79.

| Metrica | Valore |
|---|---:|
| Migliore Validation Macro-F1 | **80,95%** |
| Test Accuracy | **79,17%** |
| Test Macro-F1 | **79,31%** |
| Campioni di test | 144 |
| Parametri addestrabili | 488.582 |

Il risultato è relativamente basso rispetto:

- al numero ridotto di classi;
- alla capacità parametrica della rete;
- alla natura controllata del task.

Questo non prova che il modello sia temporalmente inadeguato, ma suggerisce che una grande capacità prevalentemente spaziale non si traduca automaticamente in una rappresentazione robusta della composizione temporale.

La validation ha inoltre mostrato oscillazioni rilevanti e il miglior checkpoint è un picco relativamente isolato. Il fenomeno deve essere verificato su più seed.

---

## 3.2 Confusion matrix e asimmetria tra sequenze inverse

Recall per classe:

| Classe | Recall |
|---|---:|
| `13` | 87,50% |
| `18` | **95,83%** |
| `31` | 87,50% |
| `38` | 75,00% |
| `81` | **50,00%** |
| `83` | 79,17% |

Il caso più evidente è la coppia inversa:

\[
18 \leftrightarrow 81.
\]

La classe `18` viene riconosciuta quasi sempre, mentre 12 dei 24 esempi della classe `81` vengono classificati come `18`.

Il modello riconosce quindi spesso quali primitive sono presenti, ma non sempre ne rappresenta correttamente la direzione temporale.

Questa anomalia non deve produrre una soluzione specifica per le classi `18/81`. Deve essere usata come sonda per studiare una proprietà generale:

> **Directional Temporal Consistency:** la rappresentazione dovrebbe comportarsi coerentemente quando la stessa composizione viene trasformata in una sequenza con ordine noto e inverso.

Le analisi future devono considerare tutte le coppie:

\[
13 \leftrightarrow 31,\qquad
18 \leftrightarrow 81,\qquad
38 \leftrightarrow 83.
\]

---

## 3.3 Perturbazioni temporali

| Condizione | Accuracy | Macro-F1 |
|---|---:|---:|
| Originale | **79,17%** | **79,31%** |
| Shuffle temporale | 38,19% | 35,86% |
| Reverse actions, target originale | 10,42% | 8,53% |
| Reverse time completo | 9,03% | 7,44% |
| Reverse actions, target rimappato | **79,17%** | **79,31%** |

L’uguaglianza esatta tra originale e `reverse_actions + reverse_class` costituisce un controllo positivo del pairing tra classi inverse.

La baseline non è temporalmente cieca:

- cambiando l’ordine delle azioni, la predizione cambia nel 73,61% dei casi;
- l’86,84% dei campioni originariamente corretti diventa errato quando la catena inversa mantiene il target originale;
- lo shuffle distrugge gran parte dell’ordine, ma preserva parte del riconoscimento del contenuto.

La conclusione corretta della Fase 1 è:

> **La baseline usa l’ordine grossolano, ma non sappiamo ancora se rappresenti dinamiche interne, tempi relativi, transizioni, velocità o memoria temporale selettiva.**

---

## 3.4 Curva sui prefissi

| Frazione osservata | Accuracy | Macro-F1 |
|---:|---:|---:|
| 25% | 42,36% | 35,81% |
| 50% | 50,69% | 45,15% |
| 75% | **86,11%** | **86,15%** |
| 100% | 79,17% | 79,31% |

Il risultato più inatteso è:

\[
86,11\% \rightarrow 79,17\%
\]

passando dal 75% al 100% della sequenza.

Osservare quattro frame addizionali peggiora la prestazione aggregata di circa 7 punti percentuali.

Questo fenomeno può dipendere da cause differenti:

- contenuto finale ambiguo o rumoroso;
- interferenza tra nuova informazione e stato precedente;
- evoluzione dello stato anche con input nullo;
- perdita dell’informazione nei layer profondi;
- sogliatura spike;
- readout tramite media temporale uniforme;
- instabilità di un singolo checkpoint;
- proprietà della generazione DVS-GC.

La Fase 2 deve discriminare queste spiegazioni senza assumerne una a priori.

---

## 3.5 Complessità e attività

| Quantità | Valore |
|---|---:|
| Parametri | 488.582 |
| Firing rate globale element-weighted | 3,40% |
| MAC per campione | 150,996 M |
| AC activity-weighted per campione | 84,119 M |
| Energia aritmetica Horowitz-style | 0,770 mJ/campione |

Circa il 66,6% dei parametri si trova nei due patch embedding. Il primo layer full precision domina i MAC e la stima energetica aritmetica.

Questo suggerisce una possibile direzione futura:

> ottenere una competenza temporale più robusta con un modello significativamente più piccolo e con meno capacità spesa nel frontend spaziale.

La Fase 2, tuttavia, non deve ancora modificare l’architettura.

---

# 4. Correzione metodologica dopo la Fase 1

Il test set della Fase 1 è stato osservato in dettaglio e ha già influenzato la formulazione delle ipotesi.

Di conseguenza:

> **Il test set order-2 usato nella Fase 1 non è più un holdout incontaminato per guidare o validare le future modifiche architetturali.**

Il run rimane valido come esperimento esplorativo, ma da questo momento deve essere classificato come:

```text
phase1_exploratory_test
```

La Fase 2 deve usare un nuovo protocollo interno alla partizione di training originale:

```text
official training partition
├── train-core
├── checkpoint-validation
└── development-audit

official test partition
└── NON UTILIZZATA nella Fase 2
```

Tutte le sequenze generate dallo stesso file sorgente devono rimanere nello stesso split.

Le classi inverse ottenute dallo stesso file, per esempio `18/file_X` e `81/file_X`, non devono essere distribuite in split differenti.

Split predefinito consigliato, configurabile:

```yaml
train_core: 0.70
checkpoint_validation: 0.15
development_audit: 0.15
group_by: source_filename
```

Il manifest degli split deve essere salvato e riusato identicamente per tutti i seed del modello.

---

# 5. Obiettivo della Fase 2

La Fase 2 è un:

> **Mechanistic Temporal Representation Audit**

La baseline deve rimanere congelata dal punto di vista architetturale.

L’obiettivo è rispondere quantitativamente alle seguenti domande:

1. Dove emerge l’informazione sul contenuto?
2. Dove emerge l’informazione sull’ordine?
3. Quanto a lungo rimane decodificabile il passato?
4. L’arrivo di nuova dinamica aggiorna o sovrascrive in modo distruttivo lo stato?
5. Membrane e spike contengono la stessa informazione?
6. Il calo 75%→100% deriva dall’input, dallo stato, dai layer profondi o dal readout?
7. La rappresentazione è robusta a trasformazioni temporali label-preserving?
8. La rappresentazione è equivarante rispetto a trasformazioni che modificano la classe in modo noto?
9. Il modello usa una memoria causale distribuita o statistiche temporali superficiali?
10. Le anomalie osservate sono robuste rispetto al seed?

La Fase 2 non deve introdurre:

- nuovi neuroni;
- hard reset;
- adaptive threshold;
- LMU;
- delayed synapses;
- temporal attention;
- JEPA;
- predictive coding;
- nuovi patch embedding;
- nuovi meccanismi di gating;
- early exit.

Sono ammesse soltanto:

- strumentazione;
- estrazione di feature;
- perturbazioni a inferenza;
- probe sul backbone congelato;
- analisi causali;
- retraining della stessa baseline per produrre checkpoint su più seed e con il nuovo split.

---

# 6. Protocollo multi-seed

La Fase 2 deve usare almeno tre seed:

```yaml
model_seeds: [42, 123, 2026]
```

Ogni seed deve:

1. usare lo stesso split manifest;
2. addestrare la stessa architettura;
3. selezionare il checkpoint esclusivamente su `checkpoint-validation`;
4. eseguire l’audit esclusivamente su `development-audit`;
5. non accedere all’official test.

Ogni risultato deve essere salvato:

- per seed;
- come media;
- come deviazione standard;
- possibilmente con intervallo bootstrap raggruppato per file sorgente.

Un fenomeno può essere dichiarato strutturale soltanto se:

- compare con lo stesso segno nella maggioranza dei seed;
- oppure viene esplicitamente classificato come seed-sensitive.

---

# 7. Preparazione dati con metadati temporali

Il package DVS-Gesture-Chain salva i frame finali, ma non espone direttamente i confini esatti delle primitive concatenate.

Per alcuni esperimenti della Fase 2 sono necessari:

- identità della primitiva attiva a ogni timestep;
- indice di transizione;
- durata di ciascun segmento;
- file sorgente;
- parametri di generazione.

L’agente deve quindi implementare una preparazione dati versionata e riproducibile per la Fase 2.

## Requisiti

Non sovrascrivere i dati della Fase 1.

Usare una root separata, per esempio:

```text
data/dvsgc_phase2_order2_v1/
```

Per ogni sequenza salvare:

```text
frames
source_filename
primitive_sequence
segment_lengths
transition_indices
generation_seed
frames_number
split_by
alpha_min
alpha_max
generator_version
```

I file sorgente devono essere ordinati deterministicamente prima della generazione.

Il manifest deve contenere:

- hash/versione del generatore;
- seed;
- lista delle classi;
- mapping classe-indice;
- lista dei campioni;
- split assegnato a ogni gruppo;
- parametri di integrazione.

Se si riusa il generatore ufficiale, il wrapper deve preservarne la logica e documentare chiaramente le differenze. Non inferire i confini a posteriori da blocchi temporali uguali.

---

# 8. Architettura software richiesta

Struttura consigliata:

```text
configs/
└── phase2_dvsgc_order2.yaml

docs/
└── phase2_mechanistic_temporal_audit.md

src/etsr/
└── phase2/
    ├── __init__.py
    ├── dataset_metadata.py
    ├── splits.py
    ├── tracing.py
    ├── prefix.py
    ├── input_audit.py
    ├── transformations.py
    ├── probes.py
    ├── causal.py
    ├── metrics.py
    ├── artifacts.py
    └── runner.py

scripts/
├── prepare_phase2.sh
├── train_phase2.sh
└── run_phase2_audit.sh

tests/
└── phase2/
    ├── test_grouped_splits.py
    ├── test_trace_equivalence.py
    ├── test_prefix_equivalence.py
    ├── test_transformations.py
    ├── test_probe_protocol.py
    ├── test_causal_influence.py
    └── test_artifact_schema.py
```

La struttura può essere adattata, ma le responsabilità devono rimanere separate.

---

# 9. Contratto CLI richiesto

Estendere la CLI con comandi espliciti:

```bash
python -m etsr.cli phase2-prepare \
    --config configs/phase2_dvsgc_order2.yaml
```

```bash
python -m etsr.cli phase2-train \
    --config configs/phase2_dvsgc_order2.yaml \
    --seed 42
```

```bash
python -m etsr.cli phase2-audit \
    --config configs/phase2_dvsgc_order2.yaml \
    --checkpoint 42=/path/seed42/best.pt \
    --checkpoint 123=/path/seed123/best.pt \
    --checkpoint 2026=/path/seed2026/best.pt
```

È accettabile un’interfaccia equivalente, ma devono essere separati:

- preparazione dati/split;
- training;
- audit multi-seed.

Ogni comando deve:

- stampare il commit Git;
- salvare la configurazione risolta;
- salvare i path reali;
- essere deterministico;
- fallire esplicitamente se gli split presentano leakage;
- rifiutare l’uso dello split `test` nella Fase 2.

---

# 10. Modulo 1 — Prefix Decision Trajectories

## 10.1 Obiettivo

Ricostruire l’evoluzione della decisione per ogni timestep:

\[
k\in\{1,\dots,T\}.
\]

Con \(T=16\), non limitarsi più ai prefissi 25%, 50%, 75%, 100%.

Per ogni campione e prefisso salvare:

\[
\hat y_k,\qquad
\ell_k,\qquad
p_k(y),\qquad
m_k.
\]

Margine corretto:

\[
m_k=
\ell_k(y)-\max_{c\neq y}\ell_k(c).
\]

## 10.2 Decomposizione esatta dell’evidenza

Il modello applica media temporale e spaziale prima della testa lineare.

Per la feature finale:

\[
z_t=\operatorname{GAP}(h_t).
\]

Il logit su un prefisso è:

\[
\ell_k=W\left(\frac{1}{k}\sum_{t=1}^{k}z_t\right)+b.
\]

Quindi il contributo per timestep è calcolabile esattamente:

\[
e_t(c)=W_c z_t.
\]

Per una coppia inversa:

\[
d_t^{AB/BA}=e_t(AB)-e_t(BA).
\]

Salvare:

- contributo di ogni timestep;
- margine cumulativo;
- timestep che supportano la classe corretta;
- timestep che supportano la classe inversa.

Implementare un test che verifichi che i logits cumulativi ricostruiti dalle feature finali coincidano con un forward esplicito sul prefisso, entro una tolleranza numerica definita.

## 10.3 Metriche

### Late Harm

\[
\operatorname{LateHarm}_{a\to b}
=P(\hat y_a=y\land\hat y_b\neq y).
\]

### Late Rescue

\[
\operatorname{LateRescue}_{a\to b}
=P(\hat y_a\neq y\land\hat y_b=y).
\]

### Net Late Utility

\[
\operatorname{NLU}_{a\to b}
=\operatorname{LateRescue}-\operatorname{LateHarm}.
\]

Calcolare almeno:

- \(12\to16\);
- tutti i passaggi \(k\to k+1\);
- massimo margine→fine sequenza.

Altre metriche:

- First Correct Time;
- Stable Decision Time;
- Prediction Flip Count;
- Margin Regression;
- Prefix Accuracy AUC;
- Prefix Macro-F1 AUC.

## 10.4 Interventi sulla coda

Per i campioni corretti a \(k=12\) e sbagliati a \(k=16\), eseguire:

1. `prefix_only`: primi 12 frame;
2. `real_tail`: sequenza completa;
3. `zero_frame_tail`: ultimi 4 frame azzerati;
4. `repeat_last_frame_tail`: ripetizione del frame 12;
5. `tail_only`: soli frame 13–16;
6. `feature_zero_tail`: azzeramento delle feature finali 13–16 prima del readout;
7. `feature_repeat_tail`: ripetizione della feature al timestep 12 nel readout.

Interpretazione:

| Risultato | Evidenza |
|---|---|
| `zero_frame_tail` peggiora | dinamica interna, BN o bias possono evolvere senza nuova evidenza |
| solo `real_tail` peggiora | contenuto finale ambiguo o rappresentato male |
| `feature_zero_tail` risolve | readout temporale responsabile |
| `tail_only` punta alla classe inversa | evidenza tardiva direzionalmente distorta |
| molti flip | decisione temporale instabile |
| decisione presa molto presto e mai aggiornata | possibile shortcut iniziale |

Usare il termine `zero-frame tail`, non “assenza assoluta di dinamica”, perché BatchNorm e altri moduli possono produrre segnali non nulli anche con frame nulli.

---

# 11. Modulo 2 — Layer-wise Temporal Representation

## 11.1 Obiettivo

Localizzare dove compaiono, persistono o si perdono:

- contenuto;
- ordine;
- informazione sul passato;
- fase temporale;
- separazione tra classi inverse.

Estrarre le uscite di:

```text
input
patch_embed1
stage1
patch_embed2
stage2
classifier
```

## 11.2 Strumentazione non invasiva

Il forward standard deve rimanere invariato.

Implementare:

- `forward()` con comportamento identico;
- una modalità diagnostica separata, per esempio `forward_with_trace()` o un recorder tramite hook;
- nessun salvataggio di trace durante il training normale;
- trace attivo soltanto sotto `torch.no_grad()`.

Per ogni layer e timestep salvare descrittori compatti:

\[
z_t^{(l)}=\operatorname{GAP}(h_t^{(l)}).
\]

Salvare anche:

- norma \(L_1\);
- norma \(L_2\);
- media;
- varianza;
- sparsità;
- firing rate;
- drift rispetto al timestep precedente.

Non salvare indiscriminatamente tutti i tensori completi per tutto il dataset.

## 11.3 Traiettoria dei centroidi

Per classe \(c\), layer \(l\) e timestep \(t\):

\[
\mu_{c,t}^{(l)}
=\mathbb{E}[z_t^{(l)}\mid y=c].
\]

Per coppie inverse:

\[
\Delta_{AB,BA}^{(l)}(t)
=\|\mu_{AB,t}^{(l)}-\mu_{BA,t}^{(l)}\|_2.
\]

Produrre:

- curve di separazione temporale;
- heatmap layer×timestep;
- confronto tra tutte le coppie inverse;
- media e deviazione standard sui seed.

## 11.4 Representation Drift

\[
\operatorname{Drift}_t^{(l)}
=\|z_t^{(l)}-z_{t-1}^{(l)}\|_2.
\]

Correlare il drift con:

- event count;
- transizioni;
- variazioni di margine;
- prediction flip;
- firing rate.

## 11.5 Membrane e spike

Estendere `MultiStepLIF` esclusivamente in modalità diagnostica per rendere accessibili:

- membrana pre-spike;
- membrana post-reset;
- spike.

Non conservare full tensors se non per un piccolo subset stratificato.

Per il dataset completo salvare versioni pooled per timestep e canale.

Possibili diagnosi:

| Osservazione | Interpretazione |
|---|---|
| ordine nella membrana ma non negli spike | perdita nella sogliatura |
| ordine in `stage1`, assente in `stage2` | perdita nei layer profondi |
| ordine in `stage2`, output errato | readout insufficiente |
| contenuto presente, ordine assente | stato temporale poco espressivo |
| asimmetria già nell’input | bias o proprietà del dato |

## 11.6 Visualizzazioni

Consentite:

- PCA delle traiettorie dei centroidi;
- heatmap layer×timestep;
- mappe spaziali medie;
- confronto appaiato tra classi inverse;
- membrane e firing rate nel tempo.

UMAP e t-SNE non devono essere usati come evidenza primaria.

---

# 12. Modulo 3 — Temporal Transformation Suite

## 12.1 Obiettivo

Misurare se il modello rappresenta proprietà temporali semanticamente utili oppure se dipende rigidamente da:

- posizione assoluta dei frame;
- durata specifica;
- numero di bin;
- densità di eventi;
- discretizzazione.

Tutte le trasformazioni devono essere deterministiche dato il seed e devono salvare i parametri applicati.

---

## 12.2 Trasformazioni label-preserving

### A. Count-preserving temporal resampling

Valutare:

\[
T\in\{8,12,16,24,32\}.
\]

La trasformazione deve:

- mantenere l’ordine;
- preservare quanto più possibile il numero totale di eventi;
- non mescolare le polarità;
- documentare gli errori numerici introdotti.

### B. Local temporal jitter

Redistribuire una frazione controllata degli eventi/frame verso bin adiacenti, preservando:

- ordine globale;
- somma totale;
- polarità.

Livelli suggeriti:

```yaml
jitter_strength: [0.05, 0.10, 0.20]
```

### C. Event dropout

Rimuovere eventi con probabilità:

```yaml
dropout_rate: [0.05, 0.10, 0.20]
```

Usare thinning deterministico rispetto al seed.

### D. Monotonic time warp

Applicare una mappatura temporale monotona che comprima o dilati porzioni della sequenza senza invertirle.

### E. Duration redistribution

Usando i confini reali salvati nei metadati, modificare la durata relativa delle primitive mantenendo:

- identità;
- ordine;
- durata totale;
- classe.

Rapporti suggeriti:

```yaml
segment_ratios:
  - [0.40, 0.60]
  - [0.50, 0.50]
  - [0.60, 0.40]
```

Questa trasformazione è specifica per DVS-Gesture-Chain, ma la proprietà studiata è generale: robustezza a variazioni di durata.

---

## 12.3 Trasformazioni label-equivariant

### Action reversal

Usare il campione accoppiato della classe inversa.

La metrica principale è:

\[
\operatorname{InverseTemporalConsistency}
=P[\hat y(Rx)=R_y(\hat y(x))].
\]

Calcolare anche:

- consistenza per coppia;
- consistenza per seed;
- matrice delle transizioni;
- margine verso la classe inversa.

---

## 12.4 Controlli distruttivi

Mantenere:

- full temporal shuffle;
- full time reversal.

Queste condizioni non sono label-preserving e non devono essere interpretate come prove dirette di comprensione semantica. Sono controlli negativi.

---

## 12.5 Metriche della suite

Per ogni intensità:

- Accuracy;
- Macro-F1;
- Content Accuracy;
- Conditional Order Accuracy;
- Prediction Consistency;
- Inverse Temporal Consistency;
- margine corretto;
- representation drift per layer;
- robustness curve;
- area sotto la robustness curve.

Distinguere sempre:

\[
\text{invarianza}
\]

da:

\[
\text{equivarianza}.
\]

---

# 13. Modulo 4 — Temporal Probes and Causal Memory Audit

## 13.1 Obiettivo

Misurare direttamente quali proprietà temporali siano linearmente accessibili negli stati interni e quanto a lungo rimangano decodificabili.

Il backbone deve essere congelato.

## 13.2 Protocollo anti-leakage dei probe

Usare:

- feature di `train-core` per addestrare il probe;
- `checkpoint-validation` per scegliere regolarizzazione e iperparametri;
- `development-audit` per la valutazione finale del probe.

Standardizzare le feature usando statistiche calcolate esclusivamente sul probe-train.

Usare modelli semplici:

- logistic regression;
- linear layer;
- ridge regression per target continui.

Aggiungere un controllo con label permutate.

---

## 13.3 Current Primitive Probe

Predire la primitiva attiva al timestep \(t\).

Misura quanto rapidamente ogni layer acquisisca la nuova dinamica.

\[
\operatorname{AcquisitionLatency}^{(l)}
=\min\{t:\operatorname{Acc}^{(l)}_{\mathrm{current}}(t)\geq\theta\}.
\]

---

## 13.4 Previous Primitive Probe

Dopo la transizione, predire la primitiva precedente.

\[
A_{\mathrm{past}}^{(l)}(\Delta t).
\]

Misura la retention del passato.

Calcolare:

- accuracy per ritardo;
- retention AUC;
- retention half-life;
- differenza tra coppie inverse.

---

## 13.5 Content Probe

Predire la coppia non ordinata:

\[
\{1,3\},\quad\{1,8\},\quad\{3,8\}.
\]

La costruzione deve derivare automaticamente dalle classi, non essere hard-coded per una singola coppia.

---

## 13.6 Order Probe

Condizionatamente alla coppia, predire l’orientamento:

\[
AB\quad\text{vs}\quad BA.
\]

Riportare:

- accuracy macro per coppia;
- accuracy condizionata al content corretto;
- performance layer-wise e timestep-wise.

---

## 13.7 Temporal Phase Probe

Predire:

- prima primitiva;
- transizione;
- seconda primitiva;
- coda.

Le etichette devono derivare dai metadati reali.

La metrica non prova comprensione semantica. Indica se la rete rappresenta il progresso temporale.

---

## 13.8 Membrane–Spike Information Gap

Applicare gli stessi probe a membrana e spike.

\[
\operatorname{MSG}^{(l)}
=
\operatorname{ProbeAcc}(u^{(l)})
-
\operatorname{ProbeAcc}(s^{(l)}).
\]

Un gap elevato indica informazione presente nella membrana ma non facilmente disponibile negli spike.

---

## 13.9 Interference Index

Misurare la perdita di informazione sul passato dopo l’arrivo della nuova dinamica:

\[
\operatorname{Interference}^{(l)}
=
A_{\mathrm{past}}^{(l)}(t_{\mathrm{before}})
-
A_{\mathrm{past}}^{(l)}(t_{\mathrm{after}}).
\]

Confrontare l’interference con l’acquisizione della primitiva corrente.

L’obiettivo generale è misurare il trade-off:

\[
\text{acquisizione del presente}
\quad\text{vs}\quad
\text{conservazione del passato}.
\]

Non introdurre reset supervisionati in corrispondenza delle transizioni.

---

## 13.10 Causal Temporal Influence Matrix

Per ogni timestep \(\tau\), mascherare localmente l’input e osservare l’effetto sulle rappresentazioni future:

\[
I_{\tau\rightarrow t}^{(l)}
=
\frac{
\|h_t^{(l)}(x)-h_t^{(l)}(x_{\setminus\tau})\|_2
}{
\|h_t^{(l)}(x)\|_2+\varepsilon
},
\qquad t\geq\tau.
\]

Calcolare anche l’effetto sul margine finale.

Per contenere il costo:

- usare un subset stratificato configurabile;
- batchare le perturbazioni;
- lavorare sotto `torch.no_grad()`.

Documentare che il masking è un intervento causale diagnostico e può essere out-of-distribution.

---

# 14. Input Audit trasversale

Prima di interpretare gli stati interni, calcolare per ogni campione e timestep:

- numero di eventi;
- eventi ON/OFF;
- polarity ratio;
- centroide spaziale;
- dispersione spaziale;
- differenza rispetto al frame precedente;
- attività cumulativa;
- posizione temporale normalizzata.

Per ogni classe e coppia inversa produrre:

- media;
- deviazione standard;
- curve temporali;
- effect size;
- confronto appaiato per file sorgente.

Implementare shortcut baselines semplici usando solo statistiche degli input:

- logistic regression su event count per timestep;
- logistic regression su attività prima/seconda parte;
- temporal centroid;
- polarity ratio;
- combinazione delle precedenti.

Training, selezione e audit devono rispettare gli stessi split.

Scopo:

> verificare se alcune anomalie siano spiegabili da statistiche semplici prima di attribuirle alla dinamica interna del modello.

---

# 15. Metriche generali da implementare

## Classificazione

- Accuracy;
- Macro-F1;
- confusion matrix;
- recall per classe.

## Content e ordine

- Content Accuracy;
- Conditional Order Accuracy;
- Inverse Temporal Consistency.

## Evidenza temporale

- Prefix Accuracy/F1 per ogni timestep;
- Prefix AUC;
- Late Harm;
- Late Rescue;
- Net Late Utility;
- First Correct Time;
- Stable Decision Time;
- Prediction Flip Count;
- Margin Regression.

## Rappresentazioni

- class centroid distance;
- inverse-pair separation;
- representation drift;
- membrane–spike gap;
- retention AUC;
- retention half-life;
- interference index.

## Robustezza

- performance per intensità;
- robustness AUC;
- consistency;
- layer-wise drift.

## Aggregazione

Ogni metrica deve essere disponibile:

- per seed;
- per classe;
- per coppia;
- globalmente;
- media ± deviazione standard sui seed.

---

# 16. Artifact richiesti

```text
artifacts/<PHASE2_ID>/
├── protocol_manifest.yaml
├── dataset_manifest.json
├── split_manifest.json
├── checkpoints_manifest.json
├── seed_42/
├── seed_123/
├── seed_2026/
├── input_audit/
│   ├── sample_metadata.csv
│   ├── temporal_statistics.csv
│   ├── paired_input_statistics.csv
│   └── shortcut_baselines.json
├── prefix_trajectories/
│   ├── prefix_metrics.csv
│   ├── sample_trajectories.npz
│   ├── evidence_contributions.npz
│   ├── late_harm.csv
│   └── tail_interventions.csv
├── representations/
│   ├── compact_features.npz
│   ├── layerwise_statistics.csv
│   ├── centroid_trajectories.npz
│   └── membrane_spike_statistics.npz
├── transformations/
│   ├── transformation_summary.csv
│   ├── robustness_curves.csv
│   ├── consistency_metrics.json
│   └── layerwise_drift.csv
├── probes/
│   ├── probe_metrics.csv
│   ├── retention_curves.csv
│   ├── interference_metrics.csv
│   └── influence_matrices.npz
└── report/
    ├── phase2_summary.json
    ├── phase2_summary.md
    └── figures/
```

Gli NPZ devono contenere metadata sufficienti per ricostruire:

- seed;
- sample index;
- source filename;
- target;
- classe;
- timestep;
- layer;
- trasformazione.

---

# 17. Rimozione completa dello smoke test

L’infrastruttura è già stata verificata sul server. Lo smoke test sintetico e il codice creato esclusivamente per esso non sono più direttamente utili al lavoro scientifico.

L’agente deve rimuovere completamente:

```text
configs/smoke.yaml
scripts/smoke_test.sh
src/etsr/data/synthetic.py
```

## Modifiche correlate obbligatorie

### `src/etsr/data/factory.py`

Rimuovere:

- import di `SyntheticTemporalOrderDataset`;
- branch `synthetic_temporal_order`;
- configurazioni e fallback esclusivamente usati dal dataset sintetico.

### `Makefile`

Rimuovere:

- target `smoke`;
- voce `smoke` da `.PHONY`;
- invocazioni a `scripts/smoke_test.sh`.

### `README.md`

Rimuovere:

- sezione smoke test;
- comandi smoke;
- riferimenti al dataset sintetico usato per verificare la pipeline.

### Documentazione

Aggiornare almeno:

```text
docs/datasets.md
docs/experiment_protocol.md
docs/repository_tree.md
docs/validation.md
```

Rimuovere i riferimenti a:

- smoke test;
- dataset sintetico;
- validazione end-to-end tramite smoke.

### Perturbazione `reverse_segments`

La perturbazione `reverse_segments` era mantenuta soltanto come approssimazione per il dataset sintetico di smoke.

Se nessun esperimento scientifico reale la usa, rimuovere:

- campo `segments` da `PerturbationSpec`;
- ramo `reverse_segments`;
- metodo `equal_temporal_chunks`;
- test `test_reverse_segments_preserves_internal_order`;
- riferimenti documentali.

Le trasformazioni segment-aware della Fase 2 devono usare i confini reali salvati nei metadati, non segmenti uguali arbitrari.

## Verifica finale della rimozione

Eseguire:

```bash
grep -RniE "smoke|synthetic_temporal_order|reverse_segments" \
    README.md Makefile configs scripts src tests docs
```

Il comando non deve trovare riferimenti residui intenzionali.

Non rimuovere i normali unit test del modello, delle metriche, delle perturbazioni reali o del profiling.

I test su piccoli tensori costruiti direttamente nel codice restano ammessi: non costituiscono smoke test.

---

# 18. Test richiesti per la Fase 2

## Split

- nessun `source_filename` in più split;
- tutte le classi inverse dello stesso file nello stesso split;
- split identico tra seed;
- official test mai caricato dai runner Phase 2.

## Tracing

- `forward()` invariato;
- tracing disattivato per default;
- logits con e senza recorder identici entro tolleranza;
- trace con shape e timestep corretti.

## Prefix

- logits cumulativi equivalenti al forward esplicito sul prefisso;
- margini e contributi ricostruiti correttamente;
- Late Harm/Rescue verificati su casi manuali.

## Trasformazioni

- determinismo rispetto al seed;
- event count preservato quando dichiarato;
- polarità non mescolate;
- ordine preservato nelle trasformazioni label-preserving;
- mapping corretto nelle trasformazioni equivariant.

## Probe

- standardizzazione fit soltanto sul train;
- iperparametri scelti soltanto sulla validation;
- audit mai usato nel fitting;
- controllo a label permutate vicino al caso casuale.

## Causal audit

- matrice triangolare causale;
- nessun effetto riportato per \(t<\tau\);
- output schema stabile.

## Artifact

- schema verificato;
- metadata completi;
- nessun path al test ufficiale;
- tutti i seed presenti.

---

# 19. Configurazione proposta

Esempio indicativo:

```yaml
experiment:
  name: phase2_dvsgc_order2_mechanistic_audit
  model_seeds: [42, 123, 2026]
  deterministic: true
  artifact_root: artifacts
  checkpoint_root: checkpoints

dataset:
  name: dvsgc_phase2
  root: data/dvsgc_phase2_order2_v1
  frames_number: 16
  sequence_length: 2
  primitive_classes: 3
  allow_consecutive_repetition: false
  alpha_min: 0.5
  alpha_max: 0.7
  generation_seed: 123
  save_segment_metadata: true

split:
  group_by: source_filename
  split_seed: 314159
  train_core: 0.70
  checkpoint_validation: 0.15
  development_audit: 0.15
  forbid_official_test: true

model:
  name: mini_qkformer
  in_channels: 2
  embed_dim: 128
  num_heads: 8
  mlp_ratio: 2.0
  lif_tau: 2.0
  lif_threshold: 1.0

training:
  epochs: 120
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.0005
  label_smoothing: 0.1
  gradient_clip_norm: 1.0
  amp: true
  select_metric: macro_f1

phase2:
  prefix:
    all_timesteps: true
    tail_start: 12
    interventions:
      - prefix_only
      - real_tail
      - zero_frame_tail
      - repeat_last_frame_tail
      - tail_only
      - feature_zero_tail
      - feature_repeat_tail

  tracing:
    layers:
      - patch_embed1
      - stage1
      - patch_embed2
      - stage2
    record_pooled_features: true
    record_lif_membrane: true
    record_lif_spikes: true
    full_tensor_samples_per_class: 4

  transformations:
    resample_timesteps: [8, 12, 16, 24, 32]
    jitter_strength: [0.05, 0.10, 0.20]
    event_dropout: [0.05, 0.10, 0.20]
    segment_ratios:
      - [0.40, 0.60]
      - [0.50, 0.50]
      - [0.60, 0.40]
    include_reverse_actions: true
    include_shuffle_control: true
    include_reverse_time_control: true

  probes:
    enabled:
      - current_primitive
      - previous_primitive
      - unordered_content
      - order
      - temporal_phase
    classifier: logistic_regression
    regularization_grid: [0.01, 0.1, 1.0, 10.0]
    shuffled_label_control: true

  causal:
    enabled: true
    samples_per_class: 8
    intervention: zero_frame
```

La configurazione finale può essere adattata in base alle risorse, ma non devono essere eliminate le domande scientifiche centrali.

---

# 20. Criteri di completamento della Fase 2

La Fase 2 è completata quando il report risponde quantitativamente a:

1. Il calo 75%→100% è riproducibile su più seed?
2. Quali campioni, classi e timestep producono Late Harm?
3. La coda reale è più dannosa di una zero-frame tail?
4. Dove emerge l’informazione sull’ordine?
5. Dove si perde o si deforma?
6. Il passato rimane decodificabile dopo la nuova dinamica?
7. Il gesto corrente viene acquisito sacrificando quello passato?
8. Membrana e spike contengono la stessa informazione?
9. Le sequenze inverse sono rappresentate in modo coerente?
10. La fragilità persiste sotto variazioni di durata e discretizzazione?
11. Semplici statistiche dell’input spiegano parte del task?
12. Quale componente rappresenta il primo collo di bottiglia causalmente supportato?

La conclusione non deve essere obbligatoriamente unica. Più cause possono coesistere.

---

# 21. Regole di interpretazione dataset-agnostic

DVS-Gesture-Chain può suggerire fenomeni, ma una conclusione deve essere formulata in termini generali.

Esempi corretti:

- **Directional temporal inconsistency**
- **Late evidence interference**
- **Short effective memory**
- **Membrane–spike information loss**
- **Order information lost in deeper layers**
- **Readout unable to preserve trajectory information**
- **Sensitivity to temporal discretization**
- **Reliance on simple temporal activity statistics**

Esempi da evitare:

- “serve un hard reset tra le gesture”;
- “il modello deve riconoscere meglio la classe 81”;
- “assegniamo più peso agli ultimi frame di questo dataset”;
- “introduciamo un meccanismo specifico per arm roll/clapping”;
- qualsiasi soluzione basata su confini supervisionati non disponibili in flussi reali.

Una futura scelta architetturale deve essere promossa soltanto se:

1. corregge una proprietà temporale generale;
2. migliora almeno due contesti o dataset;
3. non dipende da etichette di segmento disponibili solo in DVS-Gesture-Chain;
4. mantiene o migliora costo, parametri e sparsità;
5. ha una motivazione causale supportata dalla Fase 2.

---

# 22. Definition of Done per l’agente AI

Prima di dichiarare completata l’implementazione, l’agente deve:

- rimuovere tutto il codice smoke-specific;
- aggiornare documentazione e repository tree;
- implementare dataset metadata e split grouped;
- impedire l’accesso al test ufficiale;
- supportare tre seed;
- mantenere invariato il forward standard;
- implementare i quattro moduli;
- implementare input audit e shortcut baselines;
- produrre artifact secondo schema;
- aggiungere unit test;
- eseguire `pytest -q`;
- eseguire `ruff check src tests`;
- documentare comandi SMILIES/Singularity;
- aggiornare il README con una breve descrizione della Fase 2;
- annotare nel README i nuovi artifact e il loro significato;
- effettuare commit sul branch `developer`;
- non introdurre ancora alcuna modifica architetturale volta a migliorare l’accuracy.

---

# 23. Sintesi operativa finale

La Fase 1 ha dimostrato che Mini-QKFormer utilizza l’ordine grossolano ma presenta:

- performance relativamente basse;
- forte asimmetria tra alcune sequenze inverse;
- consistenza temporale incompleta;
- peggioramento con evidenza tardiva;
- costo dominato dal frontend spaziale.

La Fase 2 deve costruire un microscopio meccanicistico, non un nuovo modello.

Il risultato atteso non è un aumento di accuracy, ma una diagnosi del tipo:

> **L’informazione sul contenuto emerge nel frontend; l’ordine diventa decodificabile nello stadio intermedio; l’arrivo della seconda dinamica riduce la decodificabilità del passato; le membrane conservano più informazione degli spike; il readout medio amplifica l’interferenza tardiva.**

Oppure una diagnosi differente, purché supportata da:

- più seed;
- split senza leakage;
- analisi layer-wise;
- probe;
- trasformazioni controllate;
- interventi causali;
- artifact riproducibili.

Solo dopo questa diagnosi inizierà la progettazione incrementale di un modello più piccolo, più efficiente e capace di rappresentare meglio la dinamica temporale.
