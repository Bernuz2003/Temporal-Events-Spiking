# Protocollo DVS-Lip

**Aggiornato:** 2026-09-09

## Dataset e split

DVS-Lip contiene 100 parole e 19.871 campioni. Lo split ufficiale è 14.896 campioni da 30 speaker
per training e 4.975 da 10 speaker per test. Il rilascio locale non contiene una mappa affidabile
campione→speaker nel training; lo sviluppo usa un manifest deterministico sample-stratified da
11.901/2.995 campioni. Ogni risultato dichiara questa limitazione.

Il path di sviluppo termina in `train`. Il codice non scopre né apre il sibling `test`. L'official
test non viene usato per training, selezione, profiling o analisi degli errori.

## Rappresentazioni controllate

E0 converte ogni sample in 40 count frame OFF/ON da 128×128, finestra fisica 2 s e bin 50 ms. Gli
eventi fuori finestra causano errore; il massimo conteggio osservato è 17 e il cap uint8 è 255. I
conteggi entrano nel modello in float senza normalizzazione per la durata.

Le varianti correnti sono TBR e Spike-TBR-LIF sul controllo F. Entrambe dividono ciascun bin fisico
da 50 ms in 8 micro-bin da 6,25 ms, comprimono gli otto bit in un frame e mantengono 40 step. La
semantica canonica è polarity-agnostic, binaria per pixel/micro-bin e normalizzata per 255: conserva
l'occupazione, ma perde polarità e molteplicità. Spike-TBR applica prima un LIF per pixel con
`β=0,9`, soglia `1,1`, reset hard e reset della membrana per macro-finestra. Quest'ultima è marcata
paper-aligned perché non esiste codice ufficiale con cui verificare le scelte non completamente
specificate nel paper.

Il confronto conserva split, seed, ricetta, augmentation, F e `T=40`; cambia solo rappresentazione
e il necessario `in_channels=1`. E1 phase-count resta implementata ma sospesa. Nessuna variante usa
la durata finale o accede all'official test.

## Metriche

La selezione usa Macro-F1; si registrano accuracy, loss, confusion matrix, predizioni, Acc1/Acc2 e
curve di prefisso. La PrefixAUC standard integra l'accuracy ai prefissi
250/500/1000/1500/2000 ms e normalizza sull'intervallo 250–2000 ms. Le curve per frazione di durata
usano endpoint oracle e restano diagnostiche.

`temporal-diagnostic` aggiunge tutti i 40 punti, Macro-F1 AUC, margini/stabilità, decomposizione del
denominatore mean e attività per layer. Anche le curve allineate all'ultimo evento sono oracle; non
rappresentano una policy online né una metrica primaria.

Acc1 indica le 50 parole visivamente confondibili e Acc2 le altre 50, secondo il paper. Il manifest
`configs/dvslip_class_groups.json` rende esplicite le 25 coppie. Il codice MSTP pubblico scambia le
etichette testuali delle due parti; il repository segue la semantica del paper.

## Valutazione finale

Prima di sbloccare l'official test devono essere congelati rappresentazione, architettura, ricetta,
seed, checkpoint selection e reporting. L'accesso finale produce un artifact separato e non riapre
la selezione del modello.
