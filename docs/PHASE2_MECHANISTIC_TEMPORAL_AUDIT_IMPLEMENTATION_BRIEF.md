# Fase 2 — Mechanistic Temporal Representation Audit

## Brief operativo revisionato e specifica dell’implementazione

**Versione del protocollo:** `phase2_tdup_v1`

**Baseline:** Mini-QKFormer-like, architettura congelata

**Dataset diagnostico:** DVS-Gesture-Chain order-2

**Configurazione canonica:** `configs/phase2_dvsgc_order2.yaml`

Questo documento sostituisce la proposta estesa precedente. La revisione mantiene le domande
scientifiche essenziali, elimina misure ridondanti o premature e definisce il protocollo realmente
implementato nel repository.

La Fase 2 non cerca una diagnosi fine a se stessa. Deve produrre evidenze che permettano di decidere
se, dove e in quale modo intervenire successivamente sull’architettura per migliorare l’uso della
semantica temporale, senza ottimizzare il modello sulle anomalie di una singola coppia di classi.

---

## 1. Punto di partenza: cosa ha mostrato davvero la Fase 1

Il run esplorativo della Fase 1 ha mostrato tre fenomeni distinti:

1. il modello riconosce il contenuto delle catene meglio del loro ordine;
2. lo shuffle temporale degrada fortemente l’ordine ma conserva buona parte del contenuto;
3. nel checkpoint osservato, l’accuratezza al prefisso 12/16 supera quella a 16/16.

La fattorizzazione ricostruita sulle predizioni del run è:

| Condizione | Content Accuracy | Conditional Order Accuracy |
|---|---:|---:|
| Originale | 89,58% | 88,37% |
| Shuffle temporale | 90,97% | 41,98% |

La consistenza temporale inversa effettiva sulle coppie accoppiate è 63,89% nel complesso, con forte
eterogeneità tra coppie. L’uguaglianza aggregata tra l’accuracy originale e quella ottenuta valutando
la sequenza inversa con target rimappato non dimostra equivarianza: su un dataset bilanciato è anche
compatibile con una semplice permutazione dei campioni. La proprietà informativa è invece
campione-per-campione:

\[
\hat y(x_{BA}) = \rho(\hat y(x_{AB})),
\]

dove \(\rho\) mappa una classe nella sua inversa.

Nel singolo checkpoint della Fase 1, tra 12/16 e 16/16 si osservano 124→114 predizioni corrette, cioè
una Net Late Utility di circa −6,94 punti percentuali. Il risultato è interessante, ma non può essere
considerato strutturale prima della replica multi-seed. Il miglior checkpoint era inoltre un picco
di validation relativamente isolato.

La conclusione ammessa è quindi:

> La baseline usa l’ordine temporale grossolano, ma la rappresentazione dell’ordine è meno robusta
> del riconoscimento del contenuto e l’aggiornamento finale dell’evidenza può essere distruttivo.

Non è ancora ammesso concludere dove nasca il problema né quale modifica architetturale lo risolva.

---

## 2. Perché non viene introdotta una singola “metrica universale”

L’utilizzo della dinamica temporale non è un costrutto unidimensionale. Un modello può:

- distinguere l’ordine ma non conservarlo;
- conservare informazione decodificabile senza usarla nel readout;
- reagire alle perturbazioni temporali tramite statistiche superficiali;
- essere sensibile alla velocità pur mantenendo corretto l’ordine;
- aggiornare correttamente la decisione in media ma fallire su specifiche transizioni.

Comprimere subito queste proprietà in uno scalare richiederebbe pesi arbitrari e permetterebbe a una
dimensione forte di nasconderne una debole. La Fase 2 implementa quindi un **Temporal Dynamics
Utilization Profile (TDUP)**: un profilo standardizzato, quantitativo e trasferibile, non un indice
composito.

Le sue componenti sono:

1. fattorizzazione contenuto/ordine;
2. consistenza rispetto a trasformazioni temporali semanticamente definite;
3. utilità dell’evidenza aggiuntiva nel tempo;
4. disponibilità dell’informazione nei layer e suo uso causale;
5. controlli sulle scorciatoie presenti nell’input.

Questa è la forma di generalità scelta: non una formula legata alle classi `13/31`, ma un contratto
applicabile a qualunque dataset che esponga unità semantiche, confini o intervalli temporali,
trasformazioni con effetto noto sulla label, pairing e gruppi sorgente.

Un eventuale indice scalare potrà essere definito soltanto dopo una validazione su più dataset e una
pre-registrazione della normalizzazione e dei pesi.

---

## 3. Protocollo sperimentale e embargo dell’official test

Il test ufficiale order-2 è stato già ispezionato durante la Fase 1 e ha contribuito alla formulazione
delle ipotesi. Non è più un holdout incontaminato per lo sviluppo.

La Fase 2 usa esclusivamente la partizione ufficiale di training:

```text
official training partition
├── train_core                 70%
├── checkpoint_validation     15%
└── development_audit         15%

official test partition
└── non letta nella Fase 2
```

Lo split è deterministico, usa `split_seed: 314159` ed è raggruppato per `source_filename`. Tutte le
classi e tutte le coppie inverse derivate dallo stesso file restano nello stesso split.

Il codice impone i seguenti vincoli:

- `official_split: train`;
- `forbid_official_test: true`;
- `group_by: source_filename`;
- assenza di split chiamati `test`;
- manifest del dataset e dello split con `official_test_used: false`;
- hash dei due manifest salvati in ogni checkpoint;
- rifiuto, in audit, di checkpoint con seed, architettura o hash dei manifest differenti.

`checkpoint_validation` seleziona l’epoca. `development_audit` non seleziona né checkpoint né
iperparametri. L’official test resta riservato a una futura valutazione confermativa, dopo che
l’architettura successiva sarà stata congelata.

---

## 4. Dataset Phase 2: coppie inverse realmente controllate

Il dataset di Fase 1 non espone i confini esatti delle azioni nei tensori finali. Per una diagnosi
meccanicistica questo rende ambigue le manipolazioni locali. La Fase 2 genera perciò un dataset
separato, immutabile e versionato a partire dagli eventi della sola partizione ufficiale di training.

Configurazione canonica:

```yaml
dataset:
  root: data/dvsgc_phase2_order2_v1
  events_root: data/dvsgc/events_np/train
  frames_number: 16
  primitive_frames: 13
  primitive_ids: ['1', '3', '8']
  alpha_min: 0.5
  alpha_max: 0.7
  generation_seed: 123
```

Per una coppia di primitive \(A,B\), un file sorgente \(s\) e durate \(L_A,L_B\), il generatore
costruisce:

\[
x_{AB,s}=A_s[0:L_A] \Vert B_s[0:L_B]
\]

\[
x_{BA,s}=B_s[0:L_B] \Vert A_s[0:L_A].
\]

Le due sequenze usano quindi gli stessi chunk integrati e la durata rimane associata alla stessa
primitiva. Non vengono invertiti blocchi temporali stimati e non vengono confrontate istanze diverse.

Ogni campione salva:

- `source_filename`;
- `primitive_sequence`;
- `segment_lengths`;
- `transition_indices`;
- `generation_seed`;
- parametri d’integrazione;
- versione del generatore;
- identificatore del campione inverso.

Il generatore fallisce se:

- mancano directory o file fondamentali;
- i file non provengono da `events_np/train`;
- le primitive non condividono file sorgente;
- la forma integrata è inattesa;
- una coppia inversa manca;
- la directory di output non è vuota.

Una variazione dei parametri richiede una nuova root versionata, evitando il riuso silenzioso di un
dataset costruito con una configurazione differente.

---

## 5. Replica multi-seed

La configurazione dichiara:

```yaml
model_seeds: [42, 123, 2026]
```

Ogni seed usa:

- identica architettura;
- identico dataset e split manifest;
- identici iperparametri;
- selezione esclusiva tramite Macro-F1 su `checkpoint_validation`;
- audit esclusivo su `development_audit`.

L’audit richiede esattamente un checkpoint per ogni seed configurato. Le quantità primarie sono
salvate per seed e aggregate come media, deviazione standard e controllo di concordanza del segno.
Per la Net Late Utility viene inoltre calcolato un bootstrap raggruppato per file sorgente.

Un effetto viene considerato candidato “strutturale” soltanto se:

1. ha lo stesso segno nei seed;
2. la sua dimensione non è dominata da un solo checkpoint;
3. è coerente tra coppie di contenuto;
4. sopravvive ai controlli pertinenti.

---

## 6. Componente A — Fattorizzazione contenuto/ordine

Per target \(y\), predizione \(\hat y\) e firma non ordinata \(c(\cdot)\):

\[
ContentAcc = P(c(\hat y)=c(y))
\]

\[
ConditionalOrderAcc = P(\hat y=y \mid c(\hat y)=c(y)).
\]

Sono riportate anche Accuracy, Macro-F1, confusion matrix e risultati per coppia di contenuto.

Questa separazione risponde a due domande diverse:

- il modello riconosce quali primitive sono presenti?
- quando riconosce il contenuto, ne riconosce anche l’ordine?

La metrica non contiene riferimenti hard-coded a `18/81`. Supporta etichette composte da token
espliciti, per esempio `walk->clap`.

---

## 7. Componente B — Consistenza temporale semantica

### 7.1 Inverse Temporal Consistency

Per ogni coppia esatta \(x_{AB},x_{BA}\):

\[
ITC = P(\hat y(x_{BA})=\rho(\hat y(x_{AB}))).
\]

L’ITC è calcolata globalmente e per coppia di contenuto. È una proprietà delle predizioni accoppiate,
non l’uguaglianza tra due accuracy aggregate.

### 7.2 Duration Redistribution Consistency

Le durate dei segmenti vengono ridistribuite verso i rapporti `40:60` e `60:40` usando i confini
veri. Il rebinning preserva, per polarità e posizione spaziale, il numero totale di eventi della
sequenza. L’ordine e il contenuto rimangono invariati.

Sono misurati:

- Accuracy e Macro-F1;
- Content Accuracy;
- Conditional Order Accuracy;
- consistenza della predizione rispetto all’originale;
- errore massimo di conservazione dei conteggi.

Questa famiglia compatta sostituisce nella prima iterazione una griglia estesa di speed warp, jitter,
dropout e perturbazioni combinate. Ulteriori trasformazioni saranno aggiunte solo se il risultato
indicherà una specifica ipotesi da discriminare.

---

## 8. Componente C — Utilità dell’evidenza nel tempo

`MiniQKFormer.forward_with_trace` produce in una sola forward causale:

- feature compatte per layer e timestep;
- logit per timestep;
- logit cumulativi di ogni prefisso;
- evidenza del readout senza bias;
- bias della testa.

Il logit cumulativo a \(t\) è numericamente equivalente a una forward esplicita sul prefisso
`1..t`. Il test di equivalenza impedisce che il tracing cambi la semantica del modello.

Sono salvate curve a tutti i timestep per:

- Accuracy;
- Macro-F1;
- Content Accuracy;
- Conditional Order Accuracy;
- margine medio della classe corretta.

### 8.1 Prefix AUC raw e normalizzata

L’AUC canonica è calcolata da 4/16 a 16/16, perché i primi quattro timestep costituiscono il primo
prefisso interpretabile nel protocollo originale:

\[
AUC_{raw}=\int_{0.25}^{1} m(f)\,df
\]

\[
AUC_{norm}=\frac{AUC_{raw}}{1-0.25}.
\]

Una curva perfetta produce quindi `AUC_raw = 0.75` e `AUC_norm = 1.0`. Entrambe vengono salvate:
la prima conserva la convenzione geometrica, la seconda evita l’interpretazione errata di 0,75 come
prestazione non perfetta.

### 8.2 Late Harm, Late Rescue e Net Late Utility

Per ogni transizione tra prefissi:

- **Late Harm:** corretto prima, errato dopo;
- **Late Rescue:** errato prima, corretto dopo;
- **Net Late Utility:** `rescue_rate - harm_rate`.

La transizione primaria è 12/16→16/16. Vengono salvate anche tutte le transizioni consecutive e la
variazione del margine. La NLU primaria ha intervallo bootstrap raggruppato per `source_filename`.

Queste quantità distinguono una curva piatta dovuta ad assenza di nuova informazione da una curva
piatta che nasconde quantità simili di rescue e harm.

---

## 9. Componente D — Informazione disponibile nei layer

Le feature sono mediate soltanto nello spazio e conservano gli assi campione, tempo e canale. I layer
canonici sono:

```text
patch_embed1 → stage1 → patch_embed2 → stage2
```

Sul backbone congelato vengono addestrati probe lineari per:

- contenuto ai timestep 4, 8, 12, 16;
- ordine ai timestep 4, 8, 12, 16;
- primitiva corrente a ogni timestep;
- primitiva precedente dopo la transizione, anche in funzione del lag.

Il protocollo dei probe rispetta gli split:

- fit su `train_core`;
- selezione della regolarizzazione su `checkpoint_validation`;
- misura finale su `development_audit`;
- standardizzazione stimata soltanto sul training;
- controllo con label di training permutate.

Una maggiore accuratezza del probe in un layer profondo non dimostra che quel layer “crei”
l’informazione: profondità, dimensionalità e separabilità cambiano. I probe misurano disponibilità
lineare, non uso causale.

L’analisi membrane-vs-spike prevista nel vecchio brief è rinviata. Richiederebbe strumentazione di
tutti gli stati interni e moltiplicherebbe costo e interpretazioni prima di sapere in quali layer e
intervalli il fenomeno sia robusto.

---

## 10. Componente E — Uso causale mediante activation patching

Per ogni campione originale viene usato come donor il campione inverso dello stesso file sorgente.
Le attivazioni del donor sono acquisite al layer scelto e sostituite nell’originale soltanto nella
prima o nella seconda azione.

Poiché le due primitive possono avere durate diverse, il donor viene riallineato segmento per
segmento: ogni timestep ricevente è mappato alla stessa posizione normalizzata nel segmento donor.
Il mapping non può attraversare un confine semantico. Questo evita di patchare accidentalmente una
porzione dell’altra azione.

La traiettoria sostituita comprende anche la memoria causale accumulata dal donor fino a quel punto:
il test misura quindi l’effetto di uno stato controfattuale completo nella regione scelta, non isola
un singolo canale né una codifica “pura” della primitiva corrente.

L’effetto primario è la variazione del margine tra classe originale e classe inversa:

\[
\Delta m =
[z_y-z_{\rho(y)}]_{patched} - [z_y-z_{\rho(y)}]_{baseline}.
\]

Sono salvati, globalmente e per classe:

- media e deviazione standard di \(\Delta m\);
- tasso di cambiamento della predizione;
- tasso di cambiamento verso la classe inversa;
- layer e regione patchata.

Un’informazione è candidata a essere causalmente usata quando il probe la rende disponibile e un
patch semanticamente mirato produce un effetto coerente sul margine. Un probe alto con patch nullo
indica informazione accessibile ma non necessariamente usata dal readout.

La matrice causale completa layer×timestep×stato interno è rinviata: il primo passaggio usa soltanto
quattro layer e due regioni semantiche, con un massimo configurabile di campioni per classe.

---

## 11. Componente F — Controlli sulle scorciatoie di input

Per ogni timestep vengono salvati:

- conteggio totale e per polarità;
- rapporto di polarità;
- variazione L1 dal frame precedente;
- centroide spaziale;
- segmento semantico.

Due baseline lineari confrontano:

1. statistiche temporali complete, che conservano l’ordine dei timestep;
2. statistiche strettamente invarianti alla permutazione temporale: conteggi totali per polarità e
   media temporale dei centroidi.

Entrambe sono addestrate con lo stesso protocollo train/validation/audit e con controllo a label
permutate. Una baseline temporale forte segnala che parte del task è risolvibile tramite statistiche
di basso livello; una baseline invariante può riconoscere il contenuto, ma non dovrebbe riconoscere
l’ordine nelle coppie esatte.

---

## 12. Cosa viene deliberatamente rinviato

Non fanno parte di `phase2_tdup_v1`:

- sweep estesi di jitter, dropout, speed warp e perturbazioni composte;
- probe non lineari;
- CKA/RSA esaustiva tra ogni layer e timestep;
- tracking completo di membrane, spike e reset di ogni LIF;
- patching di ogni singolo timestep o canale;
- ablazioni del readout;
- misure di informazione mutua ad alta dimensionalità;
- un unico Temporal Utilization Score;
- modifiche architetturali.

Questi strumenti non sono rifiutati in assoluto. Sono condizionati ai risultati del core:

- se l’ordine non diventa mai decodificabile, si indagherà l’encoder/stato iniziale;
- se è decodificabile ma si perde in profondità, si localizzerà la trasformazione distruttiva;
- se rimane decodificabile ma il patching non modifica l’output, si studierà readout/gating;
- se la durata domina, si espanderanno le trasformazioni di velocità;
- se la late harm è robusta, si analizzeranno stato LIF e readout nell’intervallo finale.

---

## 13. Mappa dell’implementazione

```text
configs/phase2_dvsgc_order2.yaml

src/etsr/phase2/
├── dataset.py          generazione, manifest, split raggruppati, pairing
├── tracing.py          raccolta di logit e feature tempo-risolte
├── metrics.py          fattorizzazione, ITC, prefissi, bootstrap, aggregazione
├── transformations.py rebinning count-preserving delle durate
├── input_audit.py      statistiche e shortcut feature
├── probes.py           probe lineari e controllo shuffled-label
├── causal.py           pairing temporale e activation patching
└── runner.py           prepare, training multi-seed e audit

scripts/
├── prepare_phase2.sh
├── train_phase2_seed.sh
└── run_phase2_audit.sh
```

`src/etsr/models/mini_qkformer.py` espone `forward_with_trace` senza modificare il normale forward di
training. `src/etsr/cli.py` espone i comandi `phase2-prepare`, `phase2-train` e `phase2-audit`.

---

## 14. Artifact prodotti

La preparazione crea:

```text
data/dvsgc_phase2_order2_v1/
├── dataset_manifest.json
├── split_manifest.json
└── samples/<classe>/<source_filename>.npz
```

Ogni training crea `history.csv`, configurazione risolta, metriche di checkpoint-validation,
confusion matrix e checkpoint `best.pt`/`last.pt`. Non produce metriche sul development-audit o sul
test ufficiale.

L’audit multi-seed crea:

```text
artifacts/<PHASE2_AUDIT_ID>/
├── protocol_manifest.yaml
├── dataset_manifest.json
├── split_manifest.json
├── checkpoints_manifest.json
├── temporal_utilization_profile.json
├── phase2_summary.json
├── aggregate_content_pair.csv
├── aggregate_inverse_pair.csv
├── aggregate_prefix.csv
├── aggregate_late_update.csv
├── aggregate_transformation.csv
├── aggregate_probe.csv
├── aggregate_shortcut.csv
├── aggregate_causal.csv
└── seed_<SEED>/
    ├── seed_summary.json
    ├── prefix_metrics.csv
    ├── late_update_metrics.csv
    ├── transformation_metrics.csv
    ├── probe_metrics.csv
    ├── input_shortcut_baselines.csv
    ├── input_temporal_statistics.csv
    ├── causal_patching.csv
    └── audit_compact_traces.npz
```

Gli artifact registrano commit Git, stato dirty, hash dei checkpoint, hash dei manifest, versione del
protocollo e certificazione dell’embargo.

---

## 15. Esecuzione canonica

Preparazione una tantum:

```bash
python -m etsr.cli phase2-prepare \
  --config configs/phase2_dvsgc_order2.yaml
```

Training separato dei tre seed:

```bash
for seed in 42 123 2026; do
  python -m etsr.cli phase2-train \
    --config configs/phase2_dvsgc_order2.yaml \
    --seed "$seed"
done
```

Audit dopo aver identificato i tre `best.pt`:

```bash
python -m etsr.cli phase2-audit \
  --config configs/phase2_dvsgc_order2.yaml \
  --checkpoint 42=/path/seed42/best.pt \
  --checkpoint 123=/path/seed123/best.pt \
  --checkpoint 2026=/path/seed2026/best.pt
```

Equivalenti Make target:

```bash
make phase2-prepare
make phase2-train SEED=42
make phase2-audit CHECKPOINTS='42=/path/a.pt 123=/path/b.pt 2026=/path/c.pt'
```

I tre training vanno eseguiti in modo sequenziale sulla stessa GPU, salvo disponibilità esplicita di
GPU separate. Non occorre rigenerare il dataset tra seed.

---

## 16. Criteri di accettazione dell’implementazione

Il core è considerato tecnicamente valido quando:

- lint e compilazione statica passano;
- i test verificano split, embargo e pairing inverso;
- il forward tracciato coincide con il forward standard;
- ogni logit di prefisso coincide con una forward esplicita sullo stesso prefisso;
- il rebinning preserva i conteggi;
- l’allineamento causale non attraversa i confini;
- l’audit rifiuta checkpoint o manifest incompatibili;
- vengono forniti esattamente i checkpoint dei seed configurati;
- nessun percorso di Fase 2 costruisce o valuta l’official test.

Il core è considerato scientificamente informativo quando permette di distinguere almeno questi
quattro casi:

1. **ordine non disponibile:** probe d’ordine bassi in tutti i layer;
2. **ordine disponibile ma perso:** probe alti nei layer iniziali e bassi nei profondi;
3. **ordine disponibile ma non usato:** probe alti, patching con effetto debole;
4. **ordine usato ma aggiornato male:** ITC significativa, late harm robusta e patching finale con
   effetto distruttivo.

Le conclusioni devono rimanere condizionate al dataset diagnostico finché lo stesso profilo non sarà
replicato su almeno un secondo dataset temporalmente significativo.
