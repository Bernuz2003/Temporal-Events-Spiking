# Registro esperimenti e risultati

**Aggiornato:** 2026-09-09

Tutti i risultati DVS-Lip in questo documento provengono dalla development validation di 2.995
sample ricavata esclusivamente dall'official-train. L'official test non è stato aperto. Salvo
indicazione diversa, i run usano seed 42, 40 bin fisici da 50 ms, finestra da 2 s, mean readout a
finestra fissa e la ricetta `dvslip_e0_128`. Il checkpoint è scelto per Macro-F1 sulla stessa
validation: confronti statistici e delta sono quindi evidenza esplorativa, non una stima finale
indipendente.

## Risultati primari DVS-Lip

`ΔAcc` e `ΔF1` sono punti percentuali rispetto alla baseline B. `PrefixAUC` integra l'accuracy sui
cinque prefissi fisici 250/500/1000/1500/2000 ms e divide per l'intervallo osservato di 1,75 s.
Non è l'area da zero e non va confrontata con AUC definite su griglie diverse. Acc1 contiene le 50
classi visivamente confondibili e Acc2 le 50 parole comuni secondo il manifest versionato.

| Variante e artifact | Epoca best | Parametri | Acc % | ΔAcc pp | Macro-F1 % | ΔF1 pp | Acc1 % | Acc2 % | PrefixAUC | Peak GiB | Tempo 128 epoche |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **B** `dvslip_e0__20260825_211710__seed42` | 112 | 500.708 | 44,81 | 0,00 | 44,15 | 0,00 | 37,07 | 52,54 | 0,2453 | 4,58 | 13,09 h |
| capacity 1M `dvslip_e0_capacity_1m__20260826_100031__seed42` | 116 | 1.113.508 | 49,58 | +4,77 | 49,38 | +5,22 | 42,02 | 57,14 | 0,2932 | 6,79 | 15,85 h |
| capacity 2M `dvslip_e0_capacity_2m__20260902_120335__seed42` | 114 | 1.967.972 | 51,79 | +6,98 | 51,81 | +7,66 | 45,29 | 58,28 | 0,3365 | 9,00 | 15,18 h |
| NoCrossTime `dvslip_e0_no_cross_time__20260902_143346__seed42` | 126 | 500.708 | 15,13 | −29,68 | 13,01 | −31,14 | 12,49 | 17,76 | 0,0677 | 4,87 | 8,57 h |
| mean@last_event `dvslip_e0_readout_time_last_event__20260902_143353__seed42` | 106 | 500.708 | 42,50 | −2,30 | 42,02 | −2,13 | 36,21 | 48,80 | 0,3036 | 4,58 | 12,99 h |
| last@last_event `dvslip_e0_readout_last_readout_time_last_event__20260902_143408__seed42` | 116 | 500.708 | 32,32 | −12,49 | 31,83 | −12,32 | 26,99 | 37,65 | 0,2171 | 4,58 | 12,19 h |
| gated storico `dvslip_e0_readout_diagonal_gated__20260903_030141__seed42` | 125 | 501.476 | 19,80 | −25,01 | 17,60 | −26,55 | 17,97 | 21,63 | 0,0550 | 4,58 | 12,87 h |
| **F** `dvslip_f__20260908_094118_261095__seed42` | 123 | 431.076 | 46,88 | **+2,07** | 46,15 | **+1,99** | 37,74 | 56,01 | 0,2366 | 3,41 | 6,69 h |
| **B+TCAP** `dvslip_b_temporal_capacity__20260908_144107_967818__seed42` | 108 | 562.148 | **48,38** | **+3,57** | **48,12** | **+3,97** | **40,95** | 55,81 | 0,2611 | 4,64 | 10,34 h |
| **B+PLIF** `dvslip_b_plif__20260908_154858_510430__seed42` | 116 | 502.676 | 43,74 | −1,07 | 43,68 | −0,47 | 37,14 | 50,33 | **0,2758** | 5,62 | 13,38 h |

I controlli di capacità confermano che il task non è saturo: 1M e 2M guadagnano rispettivamente
5,22 e 7,66 punti F1, ma moltiplicano parametri, stato e costo. Non sono candidati compatti. Il
collasso NoCrossTime dimostra invece che la dipendenza fra bin è indispensabile. Il confronto dei
readout mostra che addestrare e leggere soltanto fino all'ultimo evento non migliora il punteggio
finale: mean@last_event perde 2,13 punti e last@last_event 12,32 punti.

F e TCAP sono i due risultati strutturali utili. F guadagna quasi due punti F1 riducendo parametri,
stato e proxy energetiche; TCAP guadagna quasi quattro punti e migliora entrambe le partizioni,
con il delta maggiore su Acc1. PLIF non supera B al punto operativo primario di 2 s, ma presenta
una curva temporale distinta e merita la diagnostica checkpoint-only descritta sotto.

## Gate e run interrotti

| Candidato | Bounded overfit 16×4 | Epoche | Esito | Evidenza |
|---|---:|---:|---|---|
| F | superato | 368 | full completato | ultime cinque epoche oltre soglia; loss finale 1,4784 |
| B+TCAP | superato | 138 | full completato | accuracy 98,44%, loss 1,4534 all'ultima epoca |
| B+PLIF | superato | 165 | full completato | accuracy 100%, loss 1,4782 all'ultima epoca |
| gated-v2 | fallito | 500 | full non autorizzato dal workflow | accuracy 100%, loss 1,7716; il vincolo `<1,5` non è stato abbassato |

Il full gated-v2 è stato avviato manualmente per diagnosi e fermato dopo 28 epoche. Il miglior
Macro-F1 osservato è 9,69% all'epoca 27; all'epoca 28 è 6,60%. Non è un risultato completo e non
entra nella tabella primaria. L'inizializzazione corretta ha eliminato il difetto del gated
storico, ma non ha reso efficace il readout diagonale con candidato limitato da `tanh`. Questa
forma specifica è chiusa e non riceve tuning.

## Curve temporali dei quattro confronti centrali

Queste curve sono state ottenute accorciando l'input del checkpoint addestrato sempre con 40 bin.
Sono causali, ma i punti prima di 2 s sono fuori dalla distribuzione dell'orizzonte di training e
il mean usa un denominatore diverso. Servono per diagnosticare quando emerge l'informazione, non
per dichiarare una modalità di deployment già validata.

| Tempo | B F1 % | F F1 % | TCAP F1 % | PLIF F1 % | PLIF−B pp | TCAP−B pp |
|---:|---:|---:|---:|---:|---:|---:|
| 250 ms | 1,41 | 1,08 | 0,57 | 1,66 | +0,25 | −0,84 |
| 500 ms | 5,15 | 5,29 | 3,70 | 3,93 | −1,22 | −1,45 |
| 1.000 ms | 20,94 | 16,85 | 22,62 | **27,60** | **+6,66** | +1,68 |
| 1.500 ms | 30,44 | 29,28 | 36,25 | **38,76** | **+8,32** | +5,82 |
| 2.000 ms | 44,15 | 46,15 | **48,12** | 43,68 | −0,47 | +3,97 |

| Modello | Accuracy PrefixAUC fisica | ΔAUC vs B | Accuracy AUC su frazione durata | AUC normalizzata | Endpoint |
|---|---:|---:|---:|---:|---|
| B | 0,2453 | 0,0000 | 0,1163 | 0,1292 | oracle per la sola curva relativa |
| F | 0,2366 | −0,0088 | 0,1044 | 0,1160 | oracle per la sola curva relativa |
| TCAP | 0,2611 | +0,0158 | 0,1065 | 0,1183 | oracle per la sola curva relativa |
| PLIF | **0,2758** | **+0,0305** | **0,1248** | **0,1387** | oracle per la sola curva relativa |

PLIF raggiunge prima una rappresentazione discriminativa: a 1,5 s ha già l'88,7% del proprio F1
finale, contro il 68,9% di B. Da 1,5 a 2 s PLIF non peggiora: sale da 38,76 a 43,68 (+4,93 pp).
È B a guadagnare molto di più nella coda (+13,72 pp), fino a superarlo. Sulla validation, durata
media e mediana sono 1,083 e 1,079 s; P75=1,184 s, P95=1,348 s, P99=1,457 s. Il 30,8% dei sample
è terminato entro 1 s e il 99,4% entro 1,5 s. Quindi il salto finale avviene quasi interamente
durante evoluzione ricorrente su input privo di nuovi eventi.

La curva a frazione 1,0 usa, per questo dataset, un numero di bin uguale a
`ceil(duration_us/50.000)` per tutti i 2.995 sample: è una proxy molto vicina all'ultimo evento.
PLIF raggiunge F1 31,06% contro 23,40% di B (+7,66 pp), ma la durata finale è informazione oracle e
il checkpoint non è stato addestrato a quell'orizzonte. Non giustifica `last_event+K` né una ricerca
del miglior K sul validation set.

## Profilazione hardware v4

Tutti i profili seguenti usano il best checkpoint dello stesso run e gli stessi 64 sample
validation scelti round-robin fra 64 classi. `AC attività` applica zero-skipping ideale alla densità
osservata; `AC pot.` è il potenziale binario denso. `SOP` somma AC potenziali e SOP di attention.
`MAC multivalore` comprende le convoluzioni/lineari con input non binario e, per TCAP, i suoi MAC.
Sono contatori per sample, non misure GPU o FPGA.

| Variante | Firing rate | Stato persistente | Read stato M | Write stato M | AC attività M | AC pot. M | Attn SOP M | SOP totali M | MAC multivalore M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 0,0585 | 1.099.776 | 42,89 | 45,09 | 292,07 | 5.557,45 | 11,14 | 5.568,59 | 1.530,93 |
| capacity 1M | 0,0557 | 1.648.640 | 64,30 | 67,59 | 617,66 | 12.504,27 | 24,58 | 12.528,84 | 3.161,48 |
| capacity 2M | 0,0529 | 2.197.504 | 85,70 | 90,10 | 1.042,42 | 22.229,81 | 43,25 | 22.273,06 | 5.368,73 |
| NoCrossTime | 0,0283 | 0 | 0 | 0 | 129,28 | 5.557,45 | 11,14 | 5.568,59 | 1.530,93 |
| mean@last_event | 0,0801 | 1.099.776 | 42,89 | 45,09 | 433,84 | 5.557,45 | 11,14 | 5.568,59 | 1.530,93 |
| last@last_event | 0,0986 | 1.099.776 | 42,89 | 45,09 | 535,28 | 5.557,45 | 11,14 | 5.568,59 | 1.530,93 |
| gated storico | 0,0350 | 1.099.904 | 42,90 | 45,10 | 163,81 | 5.557,45 | 11,14 | 5.568,59 | 1.530,95 |
| **F** | **0,1352** | **477.184** | **18,61** | **19,56** | 485,92 | **2.904,56** | 11,14 | **2.915,70** | **1.247,82** |
| **B+TCAP** | **0,0468** | 1.198.080 | 45,84 | 46,17 | **198,14** | 5.557,45 | 11,14 | 5.568,59 | 1.782,59 |
| **B+PLIF** | 0,0525 | 1.099.776 | 42,89 | 45,09 | 241,79 | 5.557,45 | 11,14 | 5.568,59 | 1.530,93 |

| Variante | Horowitz attività µJ/sample | Δ vs B | Horowitz densa µJ/sample | Δ vs B | Limite principale |
|---|---:|---:|---:|---:|---|
| B | 7.319,57 | 0,00% | 12.058,41 | 0,00% | riferimento FP32 parziale |
| capacity 1M | 15.127,41 | +106,67% | 25.825,36 | +114,17% | più del doppio del costo aritmetico |
| capacity 2M | 25.682,12 | +250,87% | 44.750,78 | +271,12% | non compatto |
| NoCrossTime | 7.173,06 | −2,00% | 12.058,41 | 0,00% | rimuove stato ma distrugge F1 |
| mean@last_event | 7.447,16 | +1,74% | 12.058,41 | 0,00% | il codice elabora comunque 40 bin |
| last@last_event | 7.538,46 | +2,99% | 12.058,41 | 0,00% | il codice elabora comunque 40 bin |
| gated storico | 7.204,26 | −1,58% | 12.058,53 | 0,00% | sigmoid/tanh/readout esclusi dalla proxy |
| **F** | **6.191,70** | **−15,41%** | **8.368,47** | **−30,60%** | 36,70 M confronti max-pool esclusi |
| **B+TCAP** | 8.392,66 | +14,66% | 13.216,04 | +9,60% | include 251,66 M MAC del mixer |
| **B+PLIF** | 7.274,32 | −0,62% | 12.058,41 | 0,00% | decadimenti precomputabili in inference |

La proxy Horowitz usa FP32 45 nm: 4,6 pJ/MAC, 0,9 pJ/AC e 3,7 pJ/moltiplicazione. Esclude
accessi memoria, routing, controllo, leakage, confronto/reset/integrazione LIF, pooling e riduzioni
del readout. Non rappresenta energia hardware totale. I profili non assumono quantizzazione.

### Delta dei tre candidati rispetto a B

| Metrica | F | B+TCAP | B+PLIF |
|---|---:|---:|---:|
| Parametri | −13,91% | +12,27% | +0,39% |
| Firing rate globale | +131,11% | −19,98% | −10,21% |
| AC attività | +66,37% | −32,16% | −17,21% |
| AC/SOP potenziali | −47,64% | 0,00% | 0,00% |
| MAC multivalore | −18,49% | +16,44% | 0,00% |
| Stato persistente | −56,61% | +8,94% | 0,00% |
| Letture stato | −56,61% | +6,88% | 0,00% |
| Scritture stato | −56,61% | +2,40% | 0,00% |
| Horowitz attività | −15,41% | +14,66% | −0,62% |
| Horowitz densa | −30,60% | +9,60% | 0,00% |

F non è più sparso: il firing rate più che raddoppia. Il vantaggio nasce dalla riduzione delle mappe
ad alta risoluzione, che prevale sull'aumento di attività. TCAP abbassa firing e AC osservate, ma
aggiunge MAC multivalore e buffer; il vantaggio prestazionale costa circa il 10–15% nella proxy.
PLIF riduce moderatamente firing e AC, senza vantaggio finale: la sua proprietà interessante è la
dinamica temporale, non l'efficienza al punto fisso.

## Evidenza appaiata e complementarità

Sui medesimi 2.995 sample, B ne classifica correttamente 1.342, F 1.404, TCAP 1.449 e PLIF 1.310.
F corregge 452 errori di B e ne introduce 390; TCAP corregge 488 errori di B e ne introduce 381.
Tra F e TCAP, TCAP corregge 445 errori di F e F corregge 400 errori di TCAP: le due modifiche non
producono semplicemente lo stesso insieme di successi.

I test appaiati esplorativi danno McNemar esatto p=0,0355 per F vs B, p=0,000318 per TCAP vs B,
p=0,287 per PLIF vs B e p≈0,130 per TCAP vs F. Un bootstrap stratificato per classe dà per il delta
F1 TCAP−F circa +1,97 pp con IC95% [0,02; 3,87], mentre PLIF−B attraversa zero. Poiché i checkpoint
sono selezionati su questa stessa validation e ogni variante ha un solo seed, questi numeri
stabiliscono priorità, non significatività confermativa.

## PLIF: dinamica appresa e diagnostica preregistrata

PLIF apprende scale temporali non banali. Il `stage2.attention.attn_lif` raggiunge τ medio 6,227,
range 4,820–7,086; `stage2.attention.q_lif` scende a τ medio 1,442 e
`stage2.attention.proj_lif` a 1,672. La separazione fra memoria persistente e trasformazioni rapide
è reale, ma non basta a migliorare F1 a 2 s.

Prima di qualunque nuovo training PLIF si esegue `temporal-diagnostic` sui best di B e PLIF. Il
comando produce, ogni 50 ms:

- accuracy, Macro-F1, loss, confidenza, entropia, margine top-1 e margine della classe vera;
- accordo con la decisione finale, cambi di classe e stabilità fino a 2 s;
- confronto fra `sum(h[1:t])/t` e `sum(h[1:t])/40`, per separare scala del mean e attività di coda;
- curve oracle allineate all'ultimo bin occupato per tutti gli offset fino all'orizzonte, con quota
  di sample clippati dichiarata;
- firing medio per ciascun LIF, densità dell'input e norma assoluta della feature finale, sia in
  tempo assoluto sia allineati all'ultimo evento.

L'analisi non seleziona un `K`, non cambia il checkpoint e non attribuisce risparmio computazionale
all'attuale `last_event`: il forward corrente elabora comunque tutti i bin prima della maschera.
Se PLIF mostra decisioni precoci stabili e margini robusti, il risultato motiverà in fase successiva
supervisione ai prefissi o arresto adattivo basato su confidenza, meccanismi trasferibili fra
dataset. Non motiverà una finestra `last_event+K` tarata sulla validation.

## Candidati già predisposti

**F+TCAP.** Combina i due meccanismi risultati utili senza cambiare ricetta o rappresentazione.
Ha 492.516 parametri, cioè −1,64% rispetto a B. Prima di misurare l'attività, la contabilità
strutturale stima 575.488 elementi di stato (−47,67%), circa 1.499,48 M MAC multivalore (−2,05%) e
una Horowitz densa di circa 9.526,10 µJ/sample (−21,00%). L'attività reale non è additiva e deve
essere profilata sul best del run.

**B+T.** È la compressione depthwise preregistrata del segnale TCAP: 576 coefficienti e nessun
mixing cross-channel. Non viene eseguita durante la selezione prestazionale; resta disponibile per
la fase Pareto sull'architettura congelata.

**B+E1 phase-count.** Mantiene 40 timestep fisici e divide ogni conteggio di polarità su due basi
lineari `1−phase` e `phase` all'interno del bin. La somma ricostruisce esattamente E0, ma i quattro
canali espongono il primo momento temporale sub-bin. Cambiano soltanto rappresentazione e canali del
primo conv: 501.284 parametri (+0,12% su B) e circa +377,49 M MAC se applicata a B; tutto il resto
del backbone resta a 40 passi. L'implementazione rimane disponibile, ma il full è sospeso a favore
di rappresentazioni con evidenza diretta su DVS-Lip.

**F+TBR.** TBR canonico polarity-agnostic, `N=8`, `Δt=6,25 ms`, `ΔT=50 ms`, sempre 40 macro-step.
Ha 431.004 parametri, 72 meno di F per il primo conv a un canale. Il metadata registra un
accumulatore TBR di 8 bit per pixel e le collisioni che quantificano la molteplicità scartata.

**F+Spike-TBR-LIF.** Stessa forma e stessi parametri di F+TBR, con filtro per-pixel a `β=0,9` e
soglia `1,1`. Il preprocessing richiede membrana più accumulatore; il profilo v4 del modello non
deve confonderli coi soli stati del backbone. Il metadata dichiara reset hard, reset per finestra e
assenza di codice ufficiale di riferimento.

## DVS-Gesture

`dvsgesture_e0__20260825_213021__seed42`, split speaker-disjoint sull'official-train: best epoca
117, 489.227 parametri, accuracy 84,47%, Macro-F1 83,62%, loss 0,9882. Il profilo v4 misura firing
0,02738, 13.921,48 M SOP potenziali, 3.827,30 M MAC multivalore, 1.099.776 elementi di stato e proxy
Horowitz 17.949,51 µJ attività / 30.145,89 µJ densa. Non è confrontato numericamente con DVS-Lip
perché durata, classi e protocollo differiscono.

## Regola di aggiornamento

Un full run entra nella tabella solo con `summary.json`, config risolta, ambiente, storia,
predizioni e valutazione finale. Se shortlisted richiede `hardware_profile_v4.json` ottenuto dal
proprio best. Overfit e run fermati restano separati. Per ogni nuovo candidato si riportano seed,
stato dell'official test, delta da B, PrefixAUC, operazioni, attività, stato e proxy energetiche;
le stime preventive vengono sostituite dai valori misurati appena l'artifact è disponibile.
