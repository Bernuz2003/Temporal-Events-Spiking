# Project charter

**Aggiornato:** 2026-09-07

## Obiettivo

La tesi deve ottenere e spiegare un modello spiking temporale compatto che superi in modo netto e
riproducibile la baseline MiniQKFormer da 500k parametri su DVS-Lip. Il valore scientifico deriva
dalla selezione sequenziale di modifiche motivate, da ablazioni minime e da misure con lo stesso
protocollo. Un punteggio isolato non basta.

## Ordine obbligatorio del lavoro

1. Stabilizzare una modifica **strutturale** che migliori la baseline.
2. Confermare baseline e candidata con più seed e completare il profilo hardware proxy.
3. Congelare l'architettura.
4. Applicare soltanto allora augmentation, ottimizzazione della convergenza ed eventuale
   quantizzazione, sempre con un controllo sulla baseline.
5. Usare l'official test una sola volta per la valutazione finale autorizzata.

Non si apre una griglia di clipping, learning rate, neuron model o readout durante la selezione
strutturale. Un nuovo run deve eliminare un'incertezza capace di cambiare la candidata finale.

## Criterio primario

La metrica di selezione è Macro-F1 sulla validation development. Accuracy, Acc1, Acc2, curva di
latenza causale, parametri, operazioni potenziali, attività e stato persistente sono metriche
secondarie obbligatorie. I confronti architetturali usano E0; i confronti di rappresentazione usano
F come controllo e mantengono split, seed, ricetta, augmentation, 40 macro-step e backbone.

Un miglioramento di almeno **+2 punti percentuali di Macro-F1** su seed 42 è una soglia pratica per
dare priorità alla replica, non una prova di superiorità. La decisione finale richiede più seed e
intervalli di variabilità. La candidata deve inoltre offrire un compromesso credibile tra qualità,
stato e operazioni; non è sufficiente spostare il costo fuori dal conteggio dei parametri.

## Confini

La fase corrente comprende front-end/patch embedding, memoria temporale causale compatta e il
readout già disponibile. JEPA, predictive coding, pretraining, grandi teacher, ricerca estesa di
iperparametri e quantizzazione sono rinviati. DVS-Gesture e altri dataset servono alla sola
validazione di trasferimento della candidata congelata.

## Integrità sperimentale

Ogni run deve conservare configurazione risolta, commit, stato dirty, seed, ambiente, curve,
checkpoint migliore e finale, predizioni validation e riepilogo metriche. Ogni run candidato o
baseline usato in una conclusione Pareto deve avere anche `hardware_profile_v4.json` prodotto dal suo
checkpoint selezionato. Se il checkpoint non è disponibile, il costo resta `non misurato` e non
viene interpolato da un'altra architettura.

Il test ufficiale resta escluso da sviluppo, profiling e selezione. Le affermazioni vanno formulate
come evidenza development a singolo o multiplo seed secondo quanto realmente disponibile.
