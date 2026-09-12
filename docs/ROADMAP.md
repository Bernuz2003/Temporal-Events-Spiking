# Roadmap decisiva

**Aggiornata:** 2026-09-12

## Stato del discovery

F+TCAP è il miglior modello corrente con 52,36% Macro-F1; MG-Cap raggiunge 48,42% e conferma
l'utilità della branch fine, ma con un costo molto maggiore. PLIF, gated-v2, TBR e le varianti di
clock matching sono chiusi. I risultati e i profili sono in `EXPERIMENT_LEDGER.md`.

## Iterazione corrente

1. **F+MG-Cap+TCAP, seed 42 — ultimo run combinatorio.** Unisce senza modifiche i tre moduli che
   hanno prodotto un segnale positivo. Ha 541.476 parametri; il costo del ramo fine e di TCAP si
   somma quasi interamente, quindi il run deve guadagnare almeno 2 pp F1 su F+TCAP per essere
   promosso come finalista prestazionale.
2. **Probe spaziale high-frequency — condizionale.** Se il combinato non raggiunge la soglia e
   Acc1 resta il collo di bottiglia, testare un solo mixer locale depthwise nel primo stage di
   F+TCAP, motivato congiuntamente da MaxFormer e HFR-Lip. Topologia e costo vanno chiusi prima del
   gate; nessuno sweep di kernel, posizione o larghezza.
3. **Conferma.** Il vincitore architetturale e B vengono replicati su due nuovi seed comuni e
   profilati prima del freeze.

Ogni full usa `candidate`: test statici del commit, bounded overfit 16×4, nuovo training da zero
soltanto se il gate passa, valutazione e profilo v4 del best. Nessuna modifica della ricetta è
ammessa. I full sono indipendenti sulle GPU locali `0` delle macchine fisiche. `B+T` è rinviato
perché è la compressione depthwise di TCAP; E1 phase-count è sospeso.

## Decisione alla ricezione degli artifact

| Evidenza | Decisione |
|---|---|
| MG+TCAP ≥ F+TCAP +2 pp F1 | promuovere il combinato e passare direttamente alla conferma multi-seed |
| MG+TCAP migliora meno di 2 pp | scartare MG dal finalista; F+TCAP resta riferimento e si valuta il solo probe high-frequency |
| Probe high-frequency ≥ F+TCAP +2 pp F1 | promuoverlo, quindi conferma multi-seed |
| Anche il probe high-frequency resta sotto soglia | chiudere la discovery con F+TCAP |

La soglia di 2 pp evita di promuovere un modello molto più costoso sulla base di una variazione
single-seed modesta. Non si eseguono altre varianti di tau, gain, stride o fusione MG.

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
576 coefficienti che elimina il mixing cross-channel. Le due configurazioni MultiGranular sono
chiuse prima dei run; non vengono aperti sweep di stride, larghezza o fusione.

## Stop rule

Si congela quando una modifica ulteriore non distingue un collo di bottiglia già osservato o quando
richiede uno sweep per essere definita. Non si cercano valori fini di clipping, learning rate,
numero di bin, τ, tap o soglie di early exit durante la discovery.
