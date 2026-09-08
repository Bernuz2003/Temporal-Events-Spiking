# Protocollo DVS-Lip

**Aggiornato:** 2026-09-07

## Dataset e split

DVS-Lip contiene 100 parole e 19,871 campioni. Lo split ufficiale è 14,896 campioni da 30 speaker
per training e 4,975 da 10 speaker per test. Il rilascio locale non contiene una mappa affidabile
campione→speaker nel training; lo sviluppo usa quindi un manifest deterministico sample-stratified
da 11,901/2,995 campioni. Ogni risultato deve dichiarare questa limitazione.

Il path di sviluppo termina in `train`. Il codice non scopre né apre il sibling `test`. L'official
test non viene usato per training, selezione, profiling o analisi degli errori.

## Rappresentazione congelata E0

Ogni campione diventa 40 count frame OFF/ON da 128×128, con finestra fisica di 2 s e bin da 50 ms.
Gli eventi fuori finestra sono vietati dal profilo esaustivo; il massimo conteggio osservato è 17,
quindi lo storage `uint8` con cap 255 non satura. Le feature entrano nel modello come valori di
conteggio, senza normalizzazione per la durata.

## Metriche

La selezione usa Macro-F1. Si registrano anche accuracy, loss, confusion matrix, predizioni, curve
di accuratezza per prefisso fisico causale e diagnostica per frazione relativa. Quest'ultima usa la
durata finale e viene etichettata come analisi con endpoint oracle.

Acc1 indica le 50 parole visivamente confondibili e Acc2 le altre 50, secondo il paper. Il manifest
`configs/dvslip_class_groups.json` rende esplicite le 25 coppie. Il codice MSTP pubblico scambia le
etichette testuali delle due parti; il repository segue la semantica del paper.

## Valutazione finale

Prima di sbloccare l'official test devono essere congelati architettura, ricetta, seed, checkpoint
selection e script di reporting. L'accesso finale produce un artifact separato e non riapre la
selezione del modello.
