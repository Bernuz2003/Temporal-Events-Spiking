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

### Inversione delle azioni senza confini stimati

Il package pubblico salva i frame finali ma non espone i confini esatti tra le primitive concatenate.
L'audit reale non divide quindi il tensore in blocchi temporali arbitrari: per ogni campione della
classe `AB` cerca, nella classe `BA`, il campione con lo stesso nome di file sorgente. DVS-GC genera
entrambe le catene dalle stesse istanze dei gesti, perciò la coppia contiene azioni complete in ordine
inverso.

La coppia inversa rimane una sequenza DVS-GC generata separatamente e può avere durate relative
diverse; non è una permutazione frame-esatta del tensore `AB`. L'audit fallisce esplicitamente se una
coppia manca o se la chiave classe/nome-file non è univoca. `reverse_segments` resta disponibile solo
come perturbazione approssimata per il dataset sintetico di smoke test.

## Dataset successivi

Questa release non li implementa, ma la roadmap prevede:

- DVS128 Gesture per confronto con la letteratura;
- CIFAR10-DVS per continuità con gli esperimenti precedenti;
- un dataset di azioni più ampio e naturalmente temporale per validazione realistica;
- eventualmente un dataset di natura diversa per verificare la generalità dello stato temporale appreso.

La scelta successiva sarà determinata dai risultati del Temporal Audit, non dall'intenzione di accumulare benchmark.

## Dataset diagnostico della Fase 2

La Fase 2 non riusa i frame order-2 già generati. Costruisce
`data/dvsgc_phase2_order2_v1` dagli eventi della sola partizione ufficiale di training e salva per
ogni sequenza i confini, le durate e il pairing inverso.

Per ogni file e coppia di primitive, `AB` e `BA` contengono gli stessi chunk integrati con le durate
associate alla stessa primitiva. Questo rende valide sia la ridistribuzione delle durate sia le
intervenzioni causali locali. Gli split 70/15/15 sono raggruppati per file sorgente e fissati nel
manifest. La partizione ufficiale di test non viene costruita né caricata.

Preparazione:

```bash
python -m etsr.cli phase2-prepare --config configs/phase2_dvsgc_order2.yaml
```

La directory di output è immutabile: se non è vuota il comando fallisce. Usare una nuova root
versionata per qualsiasi variazione della generazione.
