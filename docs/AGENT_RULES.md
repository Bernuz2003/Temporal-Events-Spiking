# Regole per agenti e collaboratori

1. Leggere `PROJECT_CHARTER.md`, `PROJECT_STATUS.md`, `ROADMAP.md` e `DECISIONS.md` prima di cambiare
   protocollo o architettura.
2. Non accedere all'official test DVS-Lip durante sviluppo, profiling o selezione.
3. Conservare il comportamento della baseline per default; nuove varianti devono essere attivate da
   campi espliciti nella config risolta.
4. Non avviare un full run senza i gate definiti in `TRAINING_RECIPE.md`.
5. Non introdurre sweep. Ogni run deve distinguere un'ipotesi indicata in `ROADMAP.md`.
6. Eseguire test, lint, bytecode compilation e shell checks pertinenti prima di chiudere una modifica.
7. Aggiornare `EXPERIMENT_LEDGER.md` solo da artifact completi e `PROJECT_STATUS.md` solo quando
   cambia la verità corrente. La storia resta in Git.
8. Un candidato shortlisted richiede il profilo del proprio best checkpoint. Non stimare firing
   rate o traffico di stato da un altro run.
9. Dichiarare sempre seed, split development, stato della profilazione e assenza di official-test.
10. Evitare nuovi documenti salvo che introducano un contratto durevole non coperto dagli undici
    documenti indicizzati in `README.md`.
