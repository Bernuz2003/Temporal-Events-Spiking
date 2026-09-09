# Profilazione hardware proxy

**Aggiornata:** 2026-09-09

## Contatori e semantica

`profile-checkpoint` elabora il best selezionato sullo stesso set round-robin di 64 sample validation
ed emette schema v4. Registra parametri/bit a precisione runtime, MAC multivalore, AC binari
potenziali e pesati per densità osservata, SOP di attention, firing per layer, confronti max-pool,
operazioni dei core temporali, stato persistente e traffico stimato. Per PLIF registra inoltre
numero di τ e range/medie per layer.

`binary_ac_activity_estimate` assume zero-skipping ideale sul singolo input di layer;
`binary_ac_potential` non lo assume. `sop_potential` è AC potenziale più attention SOP. Le
convoluzioni con count/phase input e i mixer TCAP restano MAC multivalore. Un modulo inizializzato
all'identità o a zero viene contato secondo il grafo che avrà dopo l'addestramento.

## Riferimento B e candidati misurati

| Variante | Parametri | Firing | Stato | AC attività M | SOP pot. M | MAC M | Horowitz attività µJ | Horowitz densa µJ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 500.708 | 0,05850 | 1.099.776 | 292,07 | 5.568,59 | 1.530,93 | 7.319,57 | 12.058,41 |
| F | 431.076 | 0,13520 | 477.184 | 485,92 | 2.915,70 | 1.247,82 | 6.191,70 | 8.368,47 |
| B+TCAP | 562.148 | 0,04682 | 1.198.080 | 198,14 | 5.568,59 | 1.782,59 | 8.392,66 | 13.216,04 |
| B+PLIF | 502.676 | 0,05253 | 1.099.776 | 241,79 | 5.568,59 | 1.530,93 | 7.274,32 | 12.058,41 |

F riduce la geometria ad alta risoluzione: il firing cresce del 131,11%, ma SOP potenziali, MAC,
stato e Horowitz densa calano rispettivamente del 47,64%, 18,49%, 56,61% e 30,60%. I max-pool
aggiungono 36,70 M confronti non inclusi nella proxy energetica. TCAP aggiunge 251,66 M MAC e
98.304 elementi di buffer; la minore attività binaria non compensa il termine multivalore nella
proxy. PLIF non aggiunge stato e il decadimento sigmoid-derived può essere precomputato dopo il
training.

La tabella completa, inclusi controlli 1M/2M, NoCrossTime e readout, è in
`EXPERIMENT_LEDGER.md`.

## Costi preventivi dei prossimi run

F+TCAP ha 492.516 parametri, 575.488 elementi di stato e circa 1.499,48 M MAC multivalore. La
Horowitz densa strutturale è circa 9.526,10 µJ/sample. Sono somme del grafo; firing, AC attività e
Horowitz attività non sono additivi e verranno sostituiti dal profilo del checkpoint.

F+TBR e F+Spike-TBR hanno 431.004 parametri, 72 meno di F, e mantengono 40 forward del backbone.
Il profiler v4 conta il modello a valle e allega i metadata della rappresentazione, ma non somma il
preprocessing ai totali del backbone. TBR richiede durante lo streaming un accumulatore da 8 bit
per pixel. La ricostruzione Spike-TBR richiede inoltre una membrana per pixel; con lo storage
runtime corrente il metadata dichiara 40 bit/pixel complessivi. Event-to-bit, decay, confronti e
reset del preprocessing devono essere riportati separatamente e non confusi con un risparmio del
modello. B+T ed E1 restano implementati ma non appartengono ai prossimi run.

## Proxy Horowitz

Il riferimento [Horowitz, ISSCC 2014](https://doi.org/10.1109/ISSCC.2014.6757323) assegna in FP32
45 nm 0,9 pJ/add, 3,7 pJ/multiply e quindi 4,6 pJ/MAC. La proxy densa usa tutti gli AC potenziali;
quella ad attività usa gli AC pesati per densità. Le SOP dell'attenzione restano dense. FIR e state
mixing compaiono una sola volta nei termini elementwise.

Sono esclusi accessi memoria, routing, controllo, leakage, integrazione/reset/confronto LIF,
max-pooling, sigmoid/tanh e riduzioni del readout. I valori sono una proxy aritmetica parziale per
sample, non energia totale, potenza, latenza o area su FPGA/GPU/ASIC.

## Completezza e comparabilità

Tutti i full DVS-Lip completati dispongono ora di profilo v4. Non usare i vecchi profili v1 nei
confronti: cambiavano campioni e classificazione binario/multivalore. Un candidato shortlisted
richiede best checkpoint proprio, schema v4, 64 sample e lo stesso
`sampling.indices_targets_sha256`. Per la tabella finale B e candidata saranno riprofilate su 200
sample. Un profilo mancante rimane mancante e non viene ricostruito da un altro run.

Il codice corrente implementa equazioni causali, ma `streaming_state_api` del modello completo è
ancora falso; i moduli FIR/TCAP espongono una transition verificata, mentre il backbone usa forward
di sequenza. `epoch_seconds` misura training e non inferenza. Qualunque affermazione sulla
piattaforma richiede precisione, memoria, mapping e misura dichiarati.
