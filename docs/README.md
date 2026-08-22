# Documentation policy

La documentazione serve a guidare il lavoro, non a replicare Git, test e artifact. Il progetto usa
due soli documenti di manutenzione ordinaria.

## Documenti scrivibili

- [`PROJECT_STATUS.md`](PROJECT_STATUS.md): stato corrente, blocchi reali e prossimo task. Si
  aggiorna solo quando cambia uno di questi elementi e sostituisce il testo obsoleto invece di
  accumulare un diario.
- [`DECISIONS.md`](DECISIONS.md): solo decisioni scientifiche o strutturali capaci di cambiare il
  lavoro futuro. Non registra refactoring ordinari, test eseguiti o dettagli già visibili nel codice.

## Documenti di riferimento

[`PROJECT_CHARTER.md`](PROJECT_CHARTER.md), [`AGENT_RULES.md`](AGENT_RULES.md),
[`ROADMAP.md`](ROADMAP.md), i documenti tematici e l'intero [`archive/`](archive/README.md) sono di
sola lettura durante l'implementazione ordinaria. Si modificano soltanto quando il loro contenuto è
esso stesso l'oggetto esplicito di una revisione approvata.

`ACTIVE_PLAN.md`, `TASKS.md`, `REPOSITORY_STATE.md` e `REPOSITORY_TREE.md` sono snapshot congelati
della transizione iniziale e sono superseduti da `PROJECT_STATUS.md`. Non vanno mantenuti allineati.

## Regole pratiche

- Non annotare ogni comando, file modificato o test: codice, Git e artifact sono l'evidenza.
- Non duplicare lo stesso fatto in più documenti.
- Aggiornare `PROJECT_STATUS.md` con poche righe soltanto a chiusura di un risultato sostanziale.
- Usare `DECISIONS.md` solo se una scelta cambia protocollo, architettura, split, obiettivo o confine
  del codebase.
- In caso di conflitto valgono, nell'ordine: charter, decisioni successive, stato corrente, codice e
  documenti di riferimento.

La fase DVS-GC resta recuperabile al tag `dvsgc-audit-complete-2026`; il suo archivio è indicizzato
in [`archive/dvsgc/README.md`](archive/dvsgc/README.md).
