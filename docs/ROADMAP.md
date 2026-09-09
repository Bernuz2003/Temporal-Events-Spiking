# Roadmap decisiva

**Aggiornata:** 2026-09-09

## Stato del discovery

La prima ondata è conclusa. F e TCAP hanno superato B; PLIF non migliora il punto operativo finale
ma anticipa l'emergere dell'informazione; gated-v2 è chiuso. I risultati e i profili completi sono
in `EXPERIMENT_LEDGER.md`.

## Iterazione corrente

1. **F+TCAP, seed 42 — candidato principale.** Verifica se front-end efficiente e mixing temporale
   ritardato sono compatibili. È il solo run che può superare direttamente il miglior 500k corrente
   combinando due segnali locali positivi.
2. **F+TBR, seed 42 — chiuso al gate.** Ha memorizzato il subset ma non ha raggiunto loss `<1,5`.
   Nessun full e nessuna variante della codifica canonica.
3. **F+Spike-TBR-LIF, seed 42 — full in corso.** Usa il LIF per-pixel con i valori DVS-Lip
   pubblicati `β=0,9` e soglia `1,1`. L'input locale risulta molto più sparso di TBR e lo snapshot
   all'epoca 38 è debole; il run termina senza aprire varianti della ricostruzione.
4. **Diagnostica B/PLIF — completata.** PLIF anticipa le decisioni tramite persistenza profonda,
   ma nessun cutoff supera B finale. Il ramo resta un risultato di latenza senza nuovo training.
5. **F+MultiGranular-Capacity, seed 42 — prova di utilità.** Conserva E0 e aggiunge una branch
   causale a 6,25 ms e 32×32, poi downsampling appreso, mixing temporale MIMO e fusione residua.
   Rimane sotto il numero di parametri di B e verifica l'ipotesi senza compressioni premature.
6. **F+MultiGranular-Lite, seed 42 — controllo Pareto.** Usa lo stesso encoder parametrico con
   griglia 16×16, mixing depthwise e somma diretta. Misura quanto del segnale sopravvive nella
   configurazione economica; un suo fallimento isolato non chiude la famiglia.

Ogni full usa `candidate`: test statici del commit, bounded overfit 16×4, nuovo training da zero
soltanto se il gate passa, valutazione e profilo v4 del best. Nessuna modifica della ricetta è
ammessa. I full sono indipendenti sulle GPU locali `0` delle macchine fisiche. `B+T` è rinviato
perché è la compressione depthwise di TCAP; E1 phase-count è sospeso.

## Decisione alla ricezione degli artifact

| Evidenza | Decisione |
|---|---|
| F+TCAP > TCAP e > F | finalista prestazionale; confrontare costo misurato con B e F |
| F+TCAP non supera TCAP | non assumere additività; TCAP resta finalista e F resta Pareto efficiente |
| F+TBR fallisce il gate | nessun full e nessuna variazione di bit/Δt; ramo canonico chiuso |
| F+Spike-TBR ≥ F +2 pp F1 | promuovere il filtro LIF e riportare stato e preprocessing |
| Spike-TBR non supera F | chiudere TBR/Spike-TBR; nessuno sweep di β, soglia o reset |
| MultiGranular-Capacity ≥ F +2 pp F1 | ipotesi confermata; confrontare Lite e poi combinare una sola configurazione col temporal core vincente |
| Capacity migliora ma Lite no | ramo utile, compressione attuale troppo aggressiva; nessuno sweep prima del freeze |
| Capacity e Lite non superano F | chiudere questa integrazione multi-granular senza sweep |
| Lite migliora almeno quanto Capacity | preferire Lite per il rapporto prestazione/costo |
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
576 coefficienti che elimina il mixing cross-channel. Le due configurazioni MultiGranular sono
chiuse prima dei run; non vengono aperti sweep di stride, larghezza o fusione.

## Stop rule

Si congela quando una modifica ulteriore non distingue un collo di bottiglia già osservato o quando
richiede uno sweep per essere definita. Non si cercano valori fini di clipping, learning rate,
numero di bin, τ, tap o soglie di early exit durante la discovery.
