# Roadmap decisiva

**Aggiornata:** 2026-09-09

## Stato del discovery

La prima ondata è conclusa. F e TCAP hanno superato B; PLIF non migliora il punto operativo finale
ma anticipa l'emergere dell'informazione; gated-v2 è chiuso. I risultati e i profili completi sono
in `EXPERIMENT_LEDGER.md`.

## Iterazione corrente: tre run ad alto ROI e una diagnostica

1. **F+TCAP, seed 42 — candidato principale.** Verifica se front-end efficiente e mixing temporale
   ritardato sono compatibili. È il solo run che può superare direttamente il miglior 500k corrente
   combinando due segnali locali positivi.
2. **F+TBR, seed 42 — rappresentazione canonica.** Usa 8 bit e micro-bin da 6,25 ms: ogni frame
   copre gli stessi 50 ms di E0 e il backbone continua a elaborare 40 step. Il confronto con F
   misura il valore dell'occupazione temporale intra-bin compressa.
3. **F+Spike-TBR-LIF, seed 42 — filtro della rappresentazione.** Aggiunge al punto 2 un LIF
   per-pixel con i valori DVS-Lip pubblicati `β=0,9` e soglia `1,1`. Il confronto TBR/Spike-TBR
   misura il valore del filtro dinamico a parità di F e forma dell'input.
4. **Diagnostica B/PLIF — zero training.** Valuta ogni bin e la coda event-aligned, separando
   denominatore del mean e attività ricorrente. Non seleziona iperparametri e non compete per il
   budget di full training dopo l'esecuzione.

Ogni full usa `candidate`: test statici del commit, bounded overfit 16×4, nuovo training da zero
soltanto se il gate passa, valutazione e profilo v4 del best. Nessuna modifica della ricetta è
ammessa. I tre full sono indipendenti e possono essere avviati sulle GPU locali `0` di tre
macchine fisiche. La quarta esegue la diagnostica breve e rimane riserva. `B+T` è rinviato perché
è la compressione depthwise di TCAP; E1 phase-count è sospeso perché è una variazione locale a ROI
inferiore rispetto alle rappresentazioni con evidenza diretta DVS-Lip.

## Decisione alla ricezione degli artifact

| Evidenza | Decisione |
|---|---|
| F+TCAP > TCAP e > F | finalista prestazionale; confrontare costo misurato con B e F |
| F+TCAP non supera TCAP | non assumere additività; TCAP resta finalista e F resta Pareto efficiente |
| F+TBR ≥ F +2 pp F1 | TBR entra nella shortlist; una sola combinazione col temporal core vincente |
| F+TBR < F +2 pp | nessuna variazione di bit/Δt; interpretare Spike-TBR prima di chiudere la famiglia |
| F+Spike-TBR ≥ F+TBR +2 pp | promuovere il filtro LIF e riportare stato e preprocessing |
| Spike-TBR non supera TBR | conservare TBR; nessuno sweep di β o soglia nella discovery |
| PLIF stabile prima ma non a 2 s | risultato latency/dynamics; rimandare prefix supervision/halting |
| PLIF non stabile o vantaggio dovuto alla scala | chiudere il ramo senza training |

La soglia di 2 pp serve a impedire combinazioni su rumore single-seed. Un candidato sotto soglia può
restare scientificamente interessante senza ricevere un altro full.

## Conferma dell'architettura

Dopo questa iterazione si selezionano al massimo due finalisti: uno prestazionale e, solo se diverso,
uno Pareto per efficienza. Per ciascun confronto confermativo:

1. eseguire baseline e candidata con due seed nuovi comuni;
2. riportare valori per seed, media e deviazione di Macro-F1/accuracy;
3. confrontare predizioni appaiate, Acc1/Acc2, confusioni e curve temporali;
4. profilare il best di ciascun seed o almeno dichiarare il piano di campionamento coerente;
5. congelare front-end, temporal core, rappresentazione e readout.

Quattro run comuni bastano per una candidata: B e candidata × due seed. Se rimangono due finalisti,
si usa prima il seed di discovery e l'evidenza Pareto per eliminarne uno; non si raddoppia
automaticamente la campagna.

## Dopo il freeze

La prima ottimizzazione confronta la medesima augmentation su baseline congelata e candidata:
temporal masking più spatial erasing già implementati. Poi, in ordine e con stop dopo il primo
fallimento: regolarizzazione/convergenza mirata, eventuale distillazione da teacher leggero,
quantizzazione e mappatura hardware. JEPA, predictive coding, pretraining e grandi teacher restano
fuori dal budget principale.

La compressione `TCAP → T depthwise` appartiene a questa fase: T è un FIR causale per-canale da
576 coefficienti che elimina il mixing cross-channel. MultiGranular-Lite entra prima del freeze
solo se TBR/Spike-TBR confermano il collo di bottiglia intra-bin e dopo una specifica chiusa di
topologia, fusione, stato e operazioni; non si traduce il principio MSTP in un ramo arbitrario.

## Stop rule

Si congela quando una modifica ulteriore non distingue un collo di bottiglia già osservato o quando
richiede uno sweep per essere definita. Non si cercano valori fini di clipping, learning rate,
numero di bin, τ, tap o soglie di early exit durante la discovery.
