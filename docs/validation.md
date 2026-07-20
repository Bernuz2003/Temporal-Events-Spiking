# Validazione del repository

La release è stata verificata prima della consegna con:

- `pytest -q`: **8 test superati**;
- `python -m compileall -q src tests`: import e sintassi Python validi;
- `bash -n scripts/*.sh`: sintassi degli script shell valida;
- smoke test end-to-end su dataset sintetico: training, validation, checkpoint, test finale, profiling, perturbation audit e prefix evaluation completati su CPU.

La suite è stata successivamente estesa con test per pairing delle azioni, fallimento su coppie
mancanti, metriche accoppiate e normalizzazione della Prefix AUC. Questi nuovi test richiedono una
nuova esecuzione di `pytest -q` nell'ambiente di progetto; i controlli statici sono riportati separatamente
nel report di audit.

## Non verificato in questo ambiente

- generazione e training sul dataset DVS-Gesture-Chain reale, perché i file DVS128 Gesture non erano disponibili nel runtime;
- build ed esecuzione effettiva del container Singularity, perché Singularity e i server SMILIES non sono accessibili dal runtime corrente;
- prestazioni numeriche rispetto al repository QKFormer ufficiale: la baseline è una riscrittura compatta con SSA corretta, non una replica bit-exact.

Questi tre punti restano gate espliciti da completare nell’ambiente server prima delle run
scientifiche definitive.

## Controlli dell’implementazione Fase 2

Sul branch `feature/phase2-temporal-utilization-audit` sono stati completati:

- `ruff check src tests`: superato;
- `python -m compileall -q src tests`: superato;
- `bash -n` sugli script Phase 2: superato;
- dry-run dei target Make Phase 2: superato.

Sono stati aggiunti test per split raggruppati, embargo del test, pairing inverso, fattorizzazione
contenuto/ordine, ITC, AUC raw/normalizzata, late harm/rescue, rebinning count-preserving,
allineamento causale e equivalenza tra tracing/prefissi e forward standard.

Nel runtime usato per questa modifica non sono installati `pytest`, NumPy o Torch. Per rispettare il
vincolo di non installare ambienti o dipendenze, la nuova suite non è stata eseguita qui. Deve essere
eseguita nel container del progetto con:

```bash
pytest -q
```

Non sono stati eseguiti preparazione DVS-GC, training multi-seed o audit numerico: sono run
scientifiche della Fase 2, non verifiche da simulare durante l’implementazione.
