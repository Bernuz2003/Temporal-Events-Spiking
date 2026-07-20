# Data directory

I dataset non sono versionati.

Per DVS-Gesture-Chain collocare i file originali DVS128 Gesture in:

```text
data/dvsgc/download/
```

La Fase 2 legge gli eventi estratti da `data/dvsgc/events_np/train/` e genera il dataset
metadata-rich separato in `data/dvsgc_phase2_order2_v1/`. La root Phase 2 è immutabile: per una
nuova configurazione usare un nuovo nome versionato.
