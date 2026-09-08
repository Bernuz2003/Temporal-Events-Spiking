# Profilazione hardware proxy

**Aggiornata:** 2026-09-08

## Cosa viene misurato

`profile-checkpoint` elabora gli stessi campioni validation e registra:

- parametri e bit alla precisione osservata;
- MAC multivalore potenziali di convoluzioni e layer lineari;
- accumuli potenzialmente binari dopo spike;
- operazioni dell'attenzione e dei moduli temporali espliciti;
- firing rate per layer;
- elementi/bit di stato persistente e letture/scritture stimate;
- metadati e hash del best checkpoint.
- stima aritmetica parziale Horowitz FP32, con termini inclusi/esclusi espliciti.

Per F il profiler deve contare anche confronti dei max-pooling. Per T deve esporre moltiplicazioni,
addizioni e buffer FIR separatamente. Per TCAP registra MAC MIMO e buffer per i ritardi; per PLIF
registra numero di parametri e distribuzione dei tau appresi per layer. Un'identità iniziale non
autorizza a dichiarare costo zero: il confronto Pareto usa il costo potenziale della struttura
addestrata.

## Baseline osservata

La baseline 500k usa 16.02 Mbit di pesi FP32 e 35.19 Mbit di membrane LIF. Il primo embedding
contiene 884,736 dei 1,099,776 elementi LIF. I MAC multivalore potenziali totali sono
1,530,933,760; `patch_embed2.proj` ne rappresenta 754,974,720. Il firing rate aggregato del profilo
è circa 0.0539.

Il preflight strutturale su tensori della forma DVS-Lip misura 7,088,386,560 operazioni affini
potenziali per la baseline e 4,152,373,760 per F. F usa 477,184 elementi di stato; F+T ne usa
542,720 e aggiunge 2,949,120 moltiplicazioni FIR, 1,966,080 addizioni FIR e 36,700,160 confronti
di pooling. Questi valori non includono un firing rate addestrato e non sostituiscono il profilo del
checkpoint.

Lo stesso preflight strutturale misura per **B+TCAP** 562,148 parametri, 1,198,080 elementi di
stato e 7,340,044,800 operazioni affini potenziali. Rispetto alla baseline sono +61,440 parametri,
+98,304 elementi di stato e +251,658,240 MAC multivalore per campione. La proxy aritmetica FP32
Horowitz densa coperta passa da 7,790.66 a 8,948.29 µJ/campione (+14.86%): l'aumento è maggiore
del +3.55% delle operazioni affini perché il nuovo termine è interamente multivalore. Per
**B+PLIF** il preflight misura 502,676 parametri (+1,968), lo stesso stato e gli stessi contatori
di operazioni della baseline. Questi valori derivano da input sintetici non addestrati: non si usa
il firing rate risultante per confronti scientifici.

## Completezza richiesta

Ogni baseline/candidata deve essere profilata dal proprio best checkpoint con schema v4 e
`--samples 64` durante discovery. La selezione round-robin per classe, con RNG locale seed 0,
evita il prefix ordinato che sovrarappresentava poche classi. DVS-Lip copre 64 classi su 100:
è uno screening, non una stima esaustiva. Per la tabella finale riprofilare baseline e candidata
con `--samples 200` (due campioni per classe, se disponibili); non servono nuovi training.
Si confrontano stesso dataset/split, campioni, ordine e `sampling.indices_targets_sha256`.
Un profilo mancante resta una cella
mancante: non si copia quello di un run con capacità o attività diversa.

Esistono soltanto i profili **v1** 500k e 1M: firing rate globale rispettivamente 0.053855 e
0.051779; binary AC activity estimate 268.67M e 572.78M per campione. Mancano 2M, NoCrossTime,
tre readout alternativi e DVS-Gesture. `profile-runs` recupera i best presenti sul server e scrive
`hardware_profile_v4.json`, conservando i file storici. Non confrontare v1 e v4 direttamente:
cambiano campioni e granularità della classificazione binario/multivalore, ora a batch unitario.
La config usata è `config_resolved.yaml` dello stesso run, inclusi gli override storici.

## Riferimento energetico Horowitz

[Horowitz, ISSCC 2014, Fig. 1.1.9](https://doi.org/10.1109/ISSCC.2014.6757323) riporta per 45 nm
FP32 0.9 pJ/add e 3.7 pJ/multiply: usiamo 4.6 pJ/MAC e 0.9 pJ/AC. Il campo
`energy_reference` contiene una proxy con AC potenziali e una con AC pesati per densità osservata
all'ingresso del singolo layer. Le SOP dell'attenzione restano potenziali; nessun firing rate globale
viene applicato indiscriminatamente. FIR e state mixing sono già nei totali elementwise e vengono
conteggiati una sola volta. I MAC del temporal channel mixer sono inclusi tra i multivalore. I
valori sono µJ/campione e **non energia totale del modello**.

Sono esclusi energia di integrazione/reset/confronto LIF, pooling, sigmoid/tanh, riduzioni del
readout, calcolo online del sigmoid PLIF e accessi memoria/routing/leakage. Il decadimento PLIF può
essere precomputato dopo il training; in quel caso il costo per timestep coincide con il LIF
fisso. I relativi contatori disponibili restano separati.
Non inferire una vittoria energetica del gated ignorando le sue non linearità. Il traffico FIR
assume un buffer circolare hardware; non misura le copie o gli accessi reali di PyTorch.

## Confine delle conclusioni

Il profiler non misura energia reale, latenza, area, costo del data movement reale, sparsity supportata
dal backend o precisione quantizzata. Le tabelle devono usare termini come `proxy`, `potenziale` e
`stato persistente`; le affermazioni su una piattaforma richiedono deployment e misura dedicati.
La causalità delle equazioni è valutata in inference/eval: durante training, BatchNorm aggrega
anche l'asse temporale. `epoch_seconds` nei log misura il training, escludendo validation/prefix
evaluation; non è un benchmark di inferenza e risente della contesa tra run sul server.
