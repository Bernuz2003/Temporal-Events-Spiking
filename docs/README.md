# Documentazione del progetto

Questa directory contiene soltanto documenti che governano il lavoro corrente. La storia rimane
in Git; snapshot, duplicati e documenti della fase DVS-Gesture/DVS-GC sono stati rimossi.

| Documento | Funzione |
|---|---|
| [`PROJECT_CHARTER.md`](PROJECT_CHARTER.md) | obiettivo, vincoli e criteri scientifici |
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | verità empirica corrente e prossimo passo |
| [`ROADMAP.md`](ROADMAP.md) | albero decisionale e budget dei run |
| [`VALIDATION_REFINEMENT_ROADMAP.md`](VALIDATION_REFINEMENT_ROADMAP.md) | conferma multi-seed, raffinamento e pretraining post-freeze |
| [`DATA_AUGMENTATION_RESEARCH.md`](DATA_AUGMENTATION_RESEARCH.md) | catalogo, evidenza e stato delle augmentation per dataset |
| [`PREDICTIVE_TEMPORAL_RESEARCH_REVIEW.md`](PREDICTIVE_TEMPORAL_RESEARCH_REVIEW.md) | valutazione critica di novità, supervisione predittiva e memoria condizionale; proposte da discutere |
| [`PREDICTIVE_TEMPORAL_ROADMAP.md`](PREDICTIVE_TEMPORAL_ROADMAP.md) | protocollo attivo pre-augmentation: ipotesi, controlli, budget, fusioni e conferma |
| [`DECISIONS.md`](DECISIONS.md) | decisioni attive che vincolano il lavoro futuro |
| [`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md) | registro compatto dei run e della profilazione |
| [`SOURCES.md`](SOURCES.md) | letteratura usata e limiti dei confronti |
| [`DVSLIP_PROTOCOL.md`](DVSLIP_PROTOCOL.md) | split, metriche ed embargo del test |
| [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md) | ricetta congelata per la selezione architetturale |
| [`HARDWARE_NOTES.md`](HARDWARE_NOTES.md) | schema e requisiti della profilazione |
| [`OPERATIONS_SMILIES.md`](OPERATIONS_SMILIES.md) | comandi riproducibili per server e checkpoint |
| [`AGENT_RULES.md`](AGENT_RULES.md) | regole minime per modificare il repository |

I fatti sperimentali vivono negli artifact. `PROJECT_STATUS.md` li sintetizza senza sostituirli.
Un cambiamento ordinario aggiorna al massimo stato, ledger e decisioni pertinenti; non crea nuovi
documenti di avanzamento.
