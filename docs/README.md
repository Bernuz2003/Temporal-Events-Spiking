# Project documentation

Questa directory separa la strategia scientifica stabile, lo stato operativo aggiornabile e la
documentazione storica. Il progetto è nella transizione dall'audit DVS-Gesture-Chain alla costruzione
DVS-Lip; nessun componente DVS-Lip è ancora implementato.

## Ordine di autorità

In caso di conflitto si applica questo ordine:

1. [`PROJECT_CHARTER.md`](PROJECT_CHARTER.md): obiettivo, vincoli, fasi e criteri scientifici;
2. [`DECISIONS.md`](DECISIONS.md): decisioni successive che modificano esplicitamente il charter;
3. [`ACTIVE_PLAN.md`](ACTIVE_PLAN.md): piano della fase corrente, senza facoltà di cambiare la strategia;
4. [`TASKS.md`](TASKS.md): registro operativo e dipendenze;
5. documenti tematici e README del repository;
6. [`archive/`](archive/README.md): evidenza storica, mai istruzione corrente.

Una discrepanza non va risolta scegliendo silenziosamente il documento più comodo: va registrata in
`DECISIONS.md` o segnalata al responsabile scientifico.

## Documenti attivi

| Documento | Responsabilità | Aggiornamento |
|---|---|---|
| [`PROJECT_CHARTER.md`](PROJECT_CHARTER.md) | costituzione scientifica completa | solo cambio di strategia approvato |
| [`AGENT_RULES.md`](AGENT_RULES.md) | regole di lavoro e criteri di stop | quando cambia il processo |
| [`ACTIVE_PLAN.md`](ACTIVE_PLAN.md) | unica fase attiva e prossimo gate | a ogni iterazione |
| [`ROADMAP.md`](ROADMAP.md) | fasi P0–P6 e gate | a ogni decisione di fase |
| [`TASKS.md`](TASKS.md) | task, dipendenze, stato e artefatti | durante il lavoro |
| [`DECISIONS.md`](DECISIONS.md) | log append-only delle decisioni | prima o insieme alla decisione |
| [`REPOSITORY_STATE.md`](REPOSITORY_STATE.md) | fatti osservati su codice e ambiente | dopo verifiche strutturali |
| [`REPOSITORY_TREE.md`](REPOSITORY_TREE.md) | mappa corrente dei file e dei confini | dopo cambi strutturali |
| [`LEGACY_BOUNDARY.md`](LEGACY_BOUNDARY.md) | confine frozen DVS-GC e regole di dipendenza | prima di spostare/rimuovere legacy |
| [`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md) | run che influenzano decisioni | dopo ogni run rilevante |
| [`SOURCES.md`](SOURCES.md) | fonti primarie, stato di lettura e claim consentiti | durante la review |
| [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md) | ricetta E0, budget di tuning e freeze | a ogni modifica della ricetta |
| [`HARDWARE_NOTES.md`](HARDWARE_NOTES.md) | contratti di costo e hardware card | prima di nuovi moduli stateful |
| [`novelty_matrix.md`](novelty_matrix.md) | confronto con il prior art | prima della selezione architetturale |
| [`dvslip_pareto.md`](dvslip_pareto.md) | Pareto DVS-Lip verificato | prima delle soglie di successo |
| [`OPERATIONS_SMILIES.md`](OPERATIONS_SMILIES.md) | procedura server/container | quando cambia l'ambiente |

## Etichette di evidenza

- **FACT**: verificato direttamente nel codice, in un artefatto o in una fonte primaria letta.
- **INFERENCE**: deduzione esplicita basata su fact indicati.
- **HYPOTHESIS**: proposizione da testare.
- **DECISION**: scelta operativa/scientifica con responsabile e reversal condition.
- **OPEN QUESTION**: informazione mancante che può cambiare una decisione.

Il testo importato dal charter non diventa automaticamente un fact bibliografico: ogni claim esterno
resta `TO VERIFY` in `SOURCES.md` finché paper e, quando rilevante, codice ufficiale non sono stati
letti.

## Regola di manutenzione

Al termine di un'iterazione:

1. aggiornare `ACTIVE_PLAN.md` e `TASKS.md`;
2. aggiungere decisioni, risultati o fonti soltanto nei rispettivi registri;
3. aggiornare `REPOSITORY_STATE.md` con comandi e limiti verificati;
4. controllare link, riferimenti a file e `git diff --check`;
5. non dichiarare completa una fase se il suo gate contiene elementi `BLOCKED` o `PENDING`.

## Archivio

La fase DVS-GC è congelata al tag `dvsgc-audit-complete-2026`. I documenti originari sono indicizzati
in [`archive/dvsgc/README.md`](archive/dvsgc/README.md); il notebook storico è indicizzato in
[`../notebooks/README.md`](../notebooks/README.md). I contenuti archiviati preservano la provenienza
ma non descrivono il piano attivo.
