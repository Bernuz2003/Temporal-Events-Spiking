# Stato corrente

**Aggiornato:** 2026-09-08

**Fase:** discovery strutturale; F in full training, gated-v2 fermato dal gate, TCAP e PLIF pronti

## Evidenza consolidata

| Esperimento | Parametri | Accuracy | Macro-F1 | Profilo checkpoint | Interpretazione |
|---|---:|---:|---:|---|---|
| baseline 500k, seed 42 | 500,708 | 44.81 | 44.15 | v1; v4 da recuperare | riferimento di sviluppo |
| capacity 1M, seed 42 | 1,113,508 | 49.58 | 49.38 | v1; v4 da recuperare | capacità utile ma costo >2× |
| capacity 2M, seed 42 | 1,967,972 | 51.79 | 51.81 | mancante | limite superiore osservato, non candidata compatta |
| NoCrossTime 500k | 500,708 | 15.13 | 13.01 | mancante | la dipendenza temporale è indispensabile |
| mean@last_event | 500,708 | 42.50 | 42.02 | mancante | il trailing silence non spiega la baseline |
| last@last_event | 500,708 | 32.32 | 31.83 | mancante | l'ultimo stato perde informazione |
| gated@fixed_window | 501,476 | 19.80 | 17.60 | mancante | init sfavorevole; test del principio non conclusivo |

Le predizioni disponibili riproducono esattamente F1 e confusion matrix registrate. I miglioramenti
500k→1M e 500k→2M sono compatibili con più capacità sul medesimo seed, ma non dimostrano robustezza
multi-seed né comparabilità con risultati official-test della letteratura. La validation è
sample-stratified perché il rilascio locale non espone identità speaker.

Nel gated readout storico `gate_input=1` e `gate_bias=0` portano, con feature non negative, a gate
iniziali che cancellano rapidamente la memoria remota. Su 40 passi a input nullo il rapporto
tra gradiente sul primo e sull'ultimo passo è `2^-39 ≈ 1.82e-12`. La nuova config esplicita inizializza
un update rate del 5%, indipendente dall'input: il rapporto è `0.95^39 ≈ 0.135`. Il risultato storico non
consente quindi di rigettare un readout ricorrente; non prova neppure che l'inizializzazione fosse
l'unica causa del fallimento.

## Costo osservato

Il profilo della baseline conta 16.02 Mbit di pesi FP32 e 35.19 Mbit di stato LIF, pari a 1,099,776
elementi. Il primo embedding contiene l'80.45% dello stato LIF. I due maggiori termini convolutivi
potenziali sono `patch_embed2.proj` (754,974,720 MAC, 49.3% dei MAC multivalore) e la prima
convoluzione (377,487,360 MAC). Il firing rate globale osservato è circa 0.0539. Questi sono proxy
riproducibili, non misure di energia o latenza su FPGA.

La profilazione è incompleta per 2M, controlli e DVS-Gesture. I checkpoint omessi localmente possono
consentirne il recupero sui server: `profile-runs` verifica quelli disponibili e segnala gli assenti.
Questo limita soltanto conclusioni hardware su quei run. Da ora un run shortlisted non
è chiuso finché il profilo del checkpoint migliore non esiste.

## Candidati attivi

**F — front-end piramidale.** Quattro convoluzioni 3×3 con canali 8→16→32→64, tre max-pooling e
shortcut 16→64 a stride 4. Mantiene la risoluzione 16×16 in ingresso allo stage transformer. Il
preflight integrato a forma DVS-Lip misura 431,076 parametri, 477,184 elementi di stato persistente
e 4.152 miliardi di operazioni affini potenziali, contro 500,708, 1,099,776 e 7.088 miliardi della
baseline: −13.9% parametri, −56.6% stato e −41.4% operazioni affini. I pool aggiungono 36.70 milioni
di confronti, contabilizzati separatamente.

**T — FIR temporale causale channel-wise.** Tre tap con identità iniziale, applicati prima del LIF
in due punti a bassa risoluzione: l'uscita del front-end e il downsampling del secondo embedding,
con dilatazioni 1 e 2. Su F aggiunge 576 coefficienti, 65,536 elementi di buffer, 2.95 milioni di
moltiplicazioni e 1.97 milioni di addizioni per campione. F+T totalizza quindi 431,652 parametri e
542,720 elementi di stato. È la versione minima e controllabile dell'idea PSN/multi-delay.

**TCAP — prova di capacità temporale cross-channel.** Nei medesimi due punti a bassa risoluzione
del backbone baseline applica
`y[t] = x[t] + W1 x[t-1] + W2 x[t-2] + W4 x[t-4]`, con una matrice completa per ritardo.
Le matrici partono da zero: il modello iniziale è funzione per funzione la baseline e l'unica
capacità aggiunta collega tempi diversi. A forma DVS-Lip aggiunge 61,440 parametri (+12.27%),
98,304 elementi di buffer (+8.94%) e 251,658,240 MAC multivalore per campione. Le operazioni affini
potenziali crescono del 3.55%; la proxy aritmetica FP32 Horowitz densa cresce del 14.86%. È un
upper-bound diagnostico: se produce segnale, il passo successivo è comprimerlo, non adottarlo
automaticamente come soluzione finale.

**PLIF per canale.** Ogni LIF apprende `1/tau = sigmoid(w)` per feature channel; nei moduli di
attenzione il parametro segue gli head dove quella è la dimensione semantica. `w=0` inizializza
esattamente `tau=2`, quindi il modello coincide con la baseline al primo forward. Aggiunge 1,968
parametri (+0.393%), nessuno stato temporale e nessuna nuova operazione per timestep nel grafo
profilato, assumendo il decadimento precomputato in inference. Il profilo del checkpoint registra
distribuzione e range dei tau appresi.

## Run del 2026-09-08

F ha superato il gate alla epoca 368; il full è partito da pesi nuovi. Lo snapshot locale arriva
alla epoca 7 e non è ancora interpretabile come risultato finale.

Gated-v2 ha raggiunto accuracy 1.0 sul subset, con traiettoria finita e senza overflow, ma non ha
superato il criterio preregistrato: nelle ultime cinque epoche la validation loss è rimasta circa
1.77 contro il vincolo stretto `<1.5` (minimo osservato 1.7643). Il full non è partito. Il gate non
viene abbassato dopo aver visto il risultato: il run dimostra separabilità del subset, ma è
compatibile con margini logit deboli e un'ottimizzazione più difficile del readout limitato da
`tanh` e dalla ricorrenza diagonale. Il solo gate non identifica causalmente quale dei due fattori
domini. Gated-v2 non riceve un secondo full durante questa fase.

## Prossimo gate

I test CPU coprono forma, causalità ed equivalenza step/sequence dei moduli temporali, backward,
equivalenza iniziale con la baseline, PLIF, init gated storica/corretta, profilazione e workflow con
fallimento del gate. I test CUDA/AMP a forma DVS-Lip richiedono SMILIES. Dopo il check dello stesso
commit sui due server liberi, avviare TCAP e PLIF tramite `candidate`: ciascun comando gestisce
bounded overfit, eventuale full da zero e profilo v4 del best. I due risultati vanno letti insieme
a parametri, stato, SOP/firing rate e proxy Horowitz; un gate superato non costituisce evidenza di
generalizzazione.
