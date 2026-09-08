# Decisioni attive

**Aggiornate:** 2026-09-08

1. DVS-Lip è il benchmark di sviluppo; l'official test è embargoed fino alla valutazione finale.
2. La baseline di riferimento è MiniQKFormer E0 da 500,708 parametri, seed 42, selezionata per
   Macro-F1. I modelli 1M e 2M sono controlli di capacità.
3. La selezione usa una sola ricetta congelata. Tuning, augmentation avanzata, distillazione e
   quantizzazione iniziano dopo il freeze architetturale.
4. I candidati prioritari sono F, front-end piramidale compatto, e T, FIR temporale causale
   channel-wise a tre tap. F viene testato per primo perché attacca il maggiore costo di stato.
5. Un incremento di +2 punti di Macro-F1 sul seed iniziale dà priorità alla replica ma non equivale
   a significatività o robustezza.
6. Parametri, stato, operazioni potenziali, firing rate e traffico di stato accompagnano ogni
   conclusione prestazionale. Il profilo usa sempre il best checkpoint dello stesso run.
7. Il risultato gated storico non rigetta il principio del readout ricorrente: l'inizializzazione
   attenua quasi completamente la memoria remota. Il rilancio gated-v2 è obbligatorio, indipendente
   da F e preceduto da overfit. Usa `gated_initial_memory_steps: 20`, input gate nullo e fixed window;
   le config senza il campo conservano l'inizializzazione storica.
8. PSN non riceve un run separato: T ne testa il nucleo utile con una dinamica causale, piccola e
   profilabile. PLIF è il primo fallback neuronale. PMSN, LMU, Mamba e GRU restano condizionali
   secondo la matrice in `SOURCES.md`.
9. DVS-Gesture e altri dataset confrontano soltanto baseline congelata e candidata finale.
10. La storia documentale rimane in Git. Non si creano snapshot, peer-review duplicate, task list
    parallele o copie del charter.
