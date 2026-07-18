# Dataset

## Primo dataset: DVS-Gesture-Chain

DVS-Gesture-Chain concatena primitive provenienti da DVS128 Gesture. Classi diverse possono contenere gli stessi gesti in ordine differente. Questo rende il dataset un banco diagnostico per verificare se un modello usa realmente l'ordine temporale.

La Fase 1 parte da:

- `sequence_length = 2`;
- `primitive_classes = 3`;
- nessuna ripetizione consecutiva;
- 16 frame complessivi.

La configurazione produce un task piccolo e interpretabile, utile per validare training, perturbazioni e metriche. Dopo la stabilizzazione è disponibile una configurazione con catene di quattro gesti.

### Installazione

Il package ufficiale espone `dvsgc.DVSGestureChain` come `DatasetFolder`. I file originali DVS128 Gesture non vengono scaricati automaticamente: devono essere collocati in:

```text
data/dvsgc/download/
```

Le dipendenze si installano con `make install` o `make install-dev`. `dvsgc==0.1.2` viene installato con `--no-deps` perché i metadata pubblicati richiedono ancora SpikingJelly 0.0.0.0.8; il repository usa e fissa SpikingJelly 0.0.0.0.14 per l'ambiente Python corrente.

Al primo avvio il package crea gli eventi estratti e i frame integrati. La generazione può richiedere tempo e spazio disco.

### Limitazione importante

Il package pubblico salva i frame finali ma non espone i confini esatti tra le primitive concatenate. La perturbazione `reverse_segments` divide quindi la sequenza in segmenti temporali approssimativamente uguali. Non va interpretata come ricostruzione perfetta di `B→A` a partire da `A→B`.

## Dataset successivi

Questa release non li implementa, ma la roadmap prevede:

- DVS128 Gesture per confronto con la letteratura;
- CIFAR10-DVS per continuità con gli esperimenti precedenti;
- un dataset di azioni più ampio e naturalmente temporale per validazione realistica;
- eventualmente un dataset di natura diversa per verificare la generalità dello stato temporale appreso.

La scelta successiva sarà determinata dai risultati del Temporal Audit, non dall'intenzione di accumulare benchmark.
