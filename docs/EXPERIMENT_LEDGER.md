# Registro esperimenti e risultati

**Aggiornato:** 2026-09-13

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
| **F+TCAP** `dvslip_f_temporal_capacity__20260909_124154_088394__seed42` | 122 | 492.516 | **52,79** | **+7,98** | **52,36** | **+8,21** | **44,42** | **61,15** | 0,2916 | 3,50 | 6,61 h |
| **F+MG-Cap** `dvslip_f_multigranular_capacity__20260909_201157_029311__seed42` | 127 | 480.036 | 48,95 | +4,14 | 48,42 | +4,27 | 41,28 | 56,61 | 0,2534 | 4,94 | 20,30 h |
| **F+MG-Cap+TCAP** `dvslip_f_multigranular_temporal_capacity__20260912_135802_779469__seed42` | 112 | 541.476 | **53,52** | **+8,71** | **53,15** | **+9,00** | **46,09** | **60,95** | **0,3113** | 5,02 | 19,17 h |

I controlli di capacità confermano che il task non è saturo: 1M e 2M guadagnano rispettivamente
5,22 e 7,66 punti F1, ma moltiplicano parametri, stato e costo. Non sono candidati compatti. Il
collasso NoCrossTime dimostra invece che la dipendenza fra bin è indispensabile. Il confronto dei
readout mostra che addestrare e leggere soltanto fino all'ultimo evento non migliora il punteggio
finale: mean@last_event perde 2,13 punti e last@last_event 12,32 punti.

F e TCAP sono i due risultati strutturali più solidi. F guadagna quasi due punti F1 riducendo
parametri, stato e proxy energetiche; TCAP guadagna 3,97 punti su B e 6,21 su F. Il combinato è il
record single-seed, ma MG aggiunge a F+TCAP soltanto 0,73 pp accuracy e 0,80 pp F1, meno della soglia
preregistrata di 2 pp. MG migliora soprattutto Acc1 (+1,67 pp), mentre Acc2 cala di 0,20 pp. PLIF
non supera B al punto operativo primario di 2 s, ma presenta una dinamica temporale utile alla fase
di raffinamento.

I guadagni di MG e TCAP sopra F non sono additivi: `46,15 + 2,27 + 6,21 = 54,63%` sarebbe
l'attesa puramente additiva, contro 53,15% osservato. L'interazione è −1,48 pp; MG conserva circa
il 35% del proprio guadagno marginale quando TCAP è già presente. I moduli condividono quindi parte
dell'informazione temporale che recuperano.

## Gate e run interrotti

| Candidato | Bounded overfit 16×4 | Epoche | Esito | Evidenza |
|---|---:|---:|---|---|
| F | superato | 368 | full completato | ultime cinque epoche oltre soglia; loss finale 1,4784 |
| B+TCAP | superato | 138 | full completato | accuracy 98,44%, loss 1,4534 all'ultima epoca |
| B+PLIF | superato | 165 | full completato | accuracy 100%, loss 1,4782 all'ultima epoca |
| gated-v2 | fallito | 500 | full non autorizzato dal workflow | accuracy 100%, loss 1,7716; il vincolo `<1,5` non è stato abbassato |
| F+TCAP | superato | 402 | full completato | 52,36% F1; profilo v4 presente |
| F+TBR | **fallito** | 500 | full bloccato | accuracy train 100% e validation 98,44%, ma loss minima validation 1,5368 e finale 1,5543 |
| F+Spike-TBR-LIF | superato | 294 | full fermato a 92/128 | best F1 15,74%; nessun summary/profilo finale |
| F+MG-Cap | superato | 317 | full completato | 48,42% F1; profilo v4 presente |
| F+MG-Cap+TCAP | superato | 314 | full completato | cinque epoche consecutive valide; 53,15% F1 e profilo v4 |
| MG clock-matched | fallito | 500 | full bloccato | loss minima 1,7349; norma gradiente media finale 115,3 |
| MG clock-matched impulse | fallito | 500 | full bloccato | max accuracy 95,31%, loss minima 1,6826; norma gradiente media finale 173,3 |

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

### Effetto temporale della combinazione finale

| Modello | F1 250 ms % | F1 500 ms % | F1 1 s % | F1 1,5 s % | F1 2 s % | F1-PrefixAUC | F1-AUC su durata |
|---|---:|---:|---:|---:|---:|---:|---:|
| F+TCAP | 0,48 | 4,00 | 25,98 | 39,49 | 52,36 | 0,2707 | 0,1179 |
| F+MG-Cap | 0,79 | 3,74 | 24,37 | 36,46 | 48,42 | 0,2239 | 0,0945 |
| **F+MG-Cap+TCAP** | **0,78** | **5,18** | **31,02** | **41,27** | **53,15** | **0,2942** | **0,1475** |
| capacity 2M | 2,01 | 10,34 | 36,93 | 43,38 | 51,81 | 0,3282 | 0,1645 |

MG aggiunto a F+TCAP anticipa l'informazione più di quanto migliori il punto finale: a 1 s il
vantaggio è +5,05 pp F1, a 1,5 s +1,79 pp e a 2 s +0,80 pp; il F1-PrefixAUC sale di 2,34 pp. Il
controllo 2M conserva però la migliore curva precoce. Il ramo fine è quindi complementare alla
memoria TCAP sul piano temporale, ma non chiude il gap di capacità con un ritorno sufficiente sul
punto operativo primario.

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
| **F+TCAP** | 0,0976 | 575.488 | 21,56 | 20,65 | 272,84 | 2.904,56 | 11,14 | 2.915,70 | 1.499,48 |
| **F+MG-Cap** | 0,0873 | 673.792 | 40,81 | 42,04 | 539,20 | 3.659,53 | 11,14 | 3.670,67 | 1.593,85 |
| **F+MG-Cap+TCAP** | 0,0630 | 772.096 | 43,76 | 43,12 | 286,66 | 3.659,53 | 11,14 | 3.670,67 | 1.845,51 |

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
| **F+TCAP** | **7.157,55** | **−2,21%** | **9.526,10** | **−21,00%** | miglior fronte prestazione/costo corrente |
| **F+MG-Cap** | 7.834,34 | +7,03% | 10.642,64 | −11,74% | branch fine a 320 step |
| **F+MG-Cap+TCAP** | 8.764,68 | +19,74% | 11.800,27 | −2,14% | record F1, ma ritorno marginale basso |

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

Il confronto decisivo usa le predizioni individuali. F+MG-Cap+TCAP e F+TCAP classificano entrambi
correttamente 1.152 sample; il combinato ne corregge 451 che F+TCAP sbaglia, ma ne perde 429 che
F+TCAP risolve. Il saldo di 22 sample dà McNemar esatto `p=0,479`. Il bootstrap appaiato
stratificato per classe stima `ΔF1=+0,795 pp`, IC95% `[−1,161; +2,705]`. Il delta non è robusto nel
singolo seed. Al contrario, aggiungere TCAP a MG produce `+4,732 pp` F1, IC95%
`[+2,834; +6,696]`, e McNemar `p=3,15e−6`: la capacità temporale è il contributo replicato fra
front-end diversi.

### Dinamica di ottimizzazione dei finalisti

| Modello | Best epoca | F1 finale % | F1 medio ultime 10 % | Train acc finale % | Norma grad. mediana | Clip fraction | Overflow AMP |
|---|---:|---:|---:|---:|---:|---:|---:|
| F+TCAP | 122 | 51,79 | 51,66 | 79,08 | 42,72 | 99,998% | 0,059% |
| F+MG-Cap | 127 | 47,47 | 47,69 | 66,57 | 78,54 | 99,761% | 0,063% |
| F+MG-Cap+TCAP | 112 | 52,13 | 51,85 | 79,09 | 49,91 | 99,992% | 0,059% |

I best tardivi di F+TCAP e MG indicano che 128 epoche possono troncare la convergenza; il combinato
raggiunge invece il massimo a 112 e poi oscilla di circa un punto. Il clipping globale a 1,0 è
attivo praticamente in ogni step, mentre gli overflow AMP sono trascurabili. Sono due variabili da
studiare dopo il freeze con un confronto singolo e preregistrato, non ragioni per reinterpretare i
delta architetturali già osservati.

## PLIF: diagnostica temporale checkpoint-only completata

PLIF apprende scale temporali non banali. Il `stage2.attention.attn_lif` raggiunge τ medio 6,227,
range 4,820–7,086; `stage2.attention.q_lif` scende a τ medio 1,442 e
`stage2.attention.proj_lif` a 1,672. La separazione fra memoria persistente e trasformazioni rapide
è reale, ma non basta a migliorare F1 a 2 s.

La diagnostica sui best di B e PLIF copre tutti i 2.995 sample validation, 40 punti a intervalli di
50 ms e tutti gli offset rispetto all'ultimo bin occupato. I checkpoint SHA-256 sono registrati in
`artifacts/dvslip_temporal_diagnostic_b_plif__20260909_v2/temporal_diagnostic_pair_summary.json`.
Il test conferma un vantaggio di dinamica precoce e rigetta l'ipotesi che il fenomeno dipenda
principalmente dal denominatore della media.

| Readout checkpoint-only | B Acc AUC | PLIF Acc AUC | Δ | B F1 AUC | PLIF F1 AUC | Δ |
|---|---:|---:|---:|---:|---:|---:|
| prefix mean `sum/t` | 0,22410 | 0,25122 | **+0,02712** | 0,20188 | 0,23661 | **+0,03473** |
| denominatore fisso `sum/40` | 0,22650 | 0,25170 | **+0,02520** | 0,20393 | 0,23771 | **+0,03378** |

Le due normalizzazioni differiscono poco e lasciano quasi invariato il vantaggio PLIF. A 1 s, per
esempio, il denominatore fisso aggiunge circa 0,39 pp F1 a entrambi. Il massimo scarto osservato ai
punti centrali resta inferiore a 0,7 pp. Il recupero tardivo di B deriva quindi dall'evoluzione
degli stati, non dalla sola riduzione di scala causata dal padding nel mean readout.

| Offset dall'ultimo evento | B F1 % | PLIF F1 % | Δ PLIF−B pp | Quota clippata a 2 s |
|---:|---:|---:|---:|---:|
| 0 ms | 23,40 | 31,06 | **+7,66** | 0,00% |
| 50 ms | 23,70 | 32,72 | **+9,03** | 0,00% |
| 100 ms | 23,77 | 33,57 | **+9,80** | 0,00% |
| 200 ms | 24,93 | 34,87 | **+9,95** | 0,03% |
| 300 ms | 26,37 | 36,33 | **+9,95** | 0,03% |
| 400 ms | 29,11 | 37,66 | **+8,56** | 0,20% |
| 500 ms | 31,89 | 39,11 | **+7,23** | 0,60% |
| 600 ms | 35,54 | 40,58 | +5,04 | 2,57% |
| 700 ms | 39,85 | 42,01 | +2,15 | 8,75% |
| 800 ms | 42,30 | 42,84 | +0,54 | 21,00% |
| 950 ms | 43,67 | 43,43 | −0,24 | 56,53% |

PLIF anticipa dunque una decisione utile, ma nessun punto event-aligned non clippato supera il F1
finale di B a 2 s, 44,15%. Il massimo di entrambe le curve appare soltanto quando la maggioranza
dei sample è già clippata all'orizzonte completo e coincide sostanzialmente col punto finale. Non
esiste un `K` nascosto che converta PLIF nel miglior classificatore; scegliere 200–300 ms sullo
stesso validation set ottimizzerebbe una latenza oracle senza battere il riferimento primario.

La traccia per layer spiega il vantaggio precoce. Dopo l'ultimo evento l'input è esattamente nullo
già da `L+50 ms`; PLIF mantiene però più attività nei blocchi profondi e meno in diversi strati
iniziali.

| Offset | `stage2.attention.attn_lif` B | PLIF | `stage2.encoded_abs_mean` B | PLIF |
|---:|---:|---:|---:|---:|
| L | 0,5117 | **0,6172** | 0,9090 | **0,9281** |
| +100 ms | 0,4130 | **0,5182** | 0,7838 | **0,8523** |
| +200 ms | 0,2546 | **0,3729** | 0,5769 | **0,7440** |
| +300 ms | 0,1120 | **0,2336** | 0,3585 | **0,5948** |
| +400 ms | 0,0399 | **0,1291** | 0,2382 | **0,4272** |
| +500 ms | 0,0215 | **0,0642** | 0,2013 | **0,2972** |
| +800 ms | **0,0171** | 0,0081 | **0,1874** | 0,1753 |

Il `stage2.attention.attn_lif` PLIF ha τ medio 6,227, mentre Q/projection apprendono scale rapide:
il modello separa filtraggio iniziale e persistenza semantica profonda. Dopo circa 600–800 ms il
vantaggio di attività si esaurisce e B recupera. Il risultato è quindi una proprietà di
**latenza/dinamica**, utile per motivare in futuro supervisione ai prefissi o arresto adattivo
basato su confidenza, non una promozione di PLIF per accuracy finale.

La diagnostica ha prodotto, ogni 50 ms:

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
Ha 492.516 parametri, cioè −1,64% rispetto a B. Il profilo del best misura 575.488 elementi di stato
(−47,67%), 1.499,48 M MAC multivalore (−2,05%), 2.915,70 M SOP potenziali (−47,64%) e una proxy
Horowitz di 7.157,55 µJ ad attività (−2,21%) / 9.526,10 µJ densa (−21,00%). È il finalista Pareto.

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
accumulatore TBR di 8 bit per pixel e le collisioni che quantificano la molteplicità scartata. Il
gate ha raggiunto accuracy 100% train e 98,44% validation, ma non loss `<1,5`: è un fallimento di
confidenza/separazione dei logit, non incapacità di memorizzare le 64 istanze. Il full resta
correttamente bloccato e non si apre uno sweep della codifica.

**F+Spike-TBR-LIF.** Stessa forma e stessi parametri di F+TBR, con filtro per-pixel a `β=0,9` e
soglia `1,1`. Il preprocessing richiede membrana più accumulatore; il profilo v4 del modello non
deve confonderli coi soli stati del backbone. Il metadata dichiara reset hard, reset per finestra e
assenza di codice ufficiale di riferimento.

**F+MultiGranular-Lite.** La topologia chiusa usa E0 invariato a `40×2×128×128` e un secondo stream
ON/OFF a `320×2×16×16`: 6,25 ms nel tempo e pooling spaziale non sovrapposto 8×8. Il ramo fine usa
Conv3×3 `2→16`, Conv1×1 `16→64`, LIF continui sull'intero sample e una Conv1d depthwise causale con
kernel/stride 8; produce 40 mappe che vengono sommate all'uscita 64×16×16 di F prima di stage 1.
Il Transformer vede sempre 40 step. La rappresentazione aumenta gli elementi di input solo del
12,5% rispetto a E0, conserva polarità e molteplicità e non usa endpoint oracle. Il modello ha
433.188 parametri, soltanto 2.112 più di F. Prima dell'attività misurata, il ramo aggiunge circa
23,59 M operazioni multivalore sul count fine e 89,13 M AC potenziali su feature spiking, oltre al
buffer causale della riduzione temporale. Questa configurazione è preregistrata; larghezza,
stride e clock ratio non ricevono sweep.

**F+MultiGranular-Capacity.** Usa lo stesso `multigranular_count_frame`, configurato con fine grid
`320×2×32×32`. Prima della fusione riduce lo spazio con un blocco residuo appreso `16→32`, proietta
a 64 canali, comprime ogni gruppo di otto micro-step con Conv1d MIMO `64→64` e fonde tramite
concat-conv residua. Ha 480.036 parametri: +48.960 su F e −20.672 rispetto a B. È la prova di
utilità della famiglia; Lite resta il confronto di compressione. Il best misura 673.792 elementi
di stato, 3.670,67 M SOP, 1.593,85 M MAC e 7.834,34 µJ nella proxy ad attività.

**F+MG-Cap+TCAP.** È il record development a seed 42: 53,52% accuracy e 53,15% F1. Rispetto a
F+TCAP aggiunge 0,80 pp F1, ma anche +9,94% parametri, +34,16% stato, +23,08% MAC, +25,89% SOP,
+22,45% energia proxy ad attività e +190% tempo di training. Non supera la soglia preregistrata e
resta un risultato esplorativo orientato anche alla latenza, non il finalista primario.

## Run incompleto Spike-TBR

Questi valori sono diagnostici e non entrano nella tabella primaria finché mancano summary,
predizioni finali e profilo del best.

| Run | Epoca snapshot | Train Acc % | Val Acc % | Val F1 % | Val loss | Grad norm | Clip fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| F+Spike-TBR-LIF | 92 | 26,62 | 14,76 | 14,30 | 3,5397 | 43,29 | 1,00 |

Il best F1 osservato prima dello stop è 15,74% all'epoca 82; a 92 epoche la curva resta al 14,30%.
Non è un problema numerico: loss e gradienti sono finiti e la clip fraction 1,0 compare anche negli
altri full. Il collo di bottiglia è informativo. Su 64 sample validation deterministici, Spike-TBR
emette in media 808,5
voxel macro non nulli contro 7.594,6 di TBR, rapporto 0,0982. Il reset della membrana ogni 50 ms
interrompe inoltre accumuli sub-soglia; mantenendo la stessa ricorrenza continua per il sample il
numero medio di voxel emessi sale a circa 2.123,7, cioè 2,89×. Questa misura spiega il ritardo ma
non autorizza un secondo full: la continuità fra macro-finestre non è determinata dal paper e
diventerebbe una variante locale.

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
