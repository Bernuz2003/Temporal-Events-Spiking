# Validazione della release

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

Questi tre punti sono gate espliciti da completare sul branch `developer` prima di avviare le run scientifiche definitive.
