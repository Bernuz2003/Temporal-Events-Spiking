# Decisioni attive

**Aggiornate:** 2026-09-08

1. DVS-Lip è il benchmark di sviluppo; l'official test è embargoed fino alla valutazione finale.
2. La baseline di riferimento è MiniQKFormer E0 da 500,708 parametri, seed 42, selezionata per
   Macro-F1. I modelli 1M e 2M sono controlli di capacità.
3. La selezione usa una sola ricetta congelata. Tuning, augmentation avanzata, distillazione e
   quantizzazione iniziano dopo il freeze architetturale.
4. F resta la sonda del front-end. L'ipotesi temporale esplicita segue due stadi: prima un capacity
   probe MIMO causale a ritardi 1/2/4 sulla baseline; soltanto dopo un segnale positivo si comprime
   verso il FIR depthwise T da 576 coefficienti. Un fallimento di T non rigetta da solo l'ipotesi.
5. Un incremento di +2 punti di Macro-F1 sul seed iniziale dà priorità alla replica ma non equivale
   a significatività o robustezza.
6. Parametri, stato, operazioni potenziali, firing rate e traffico di stato accompagnano ogni
   conclusione prestazionale. Il profilo usa sempre il best checkpoint dello stesso run.
7. Il risultato gated storico non rigetta il principio del readout ricorrente: l'inizializzazione
   attenua quasi completamente la memoria remota. Il rilancio gated-v2 è obbligatorio, indipendente
   da F e preceduto da overfit. Usa `gated_initial_memory_steps: 20`, input gate nullo e fixed window;
   le config senza il campo conservano l'inizializzazione storica.
8. PLIF per feature channel/head viene testato sulla baseline in parallelo al capacity probe. Parte
   esattamente da τ=2 in tutti i LIF e non cambia soglia, reset, surrogate o recipe. I due run
   rispondono a ipotesi diverse e non vengono combinati prima di osservarne gli esiti. PMSN, LMU,
   Mamba e GRU restano condizionali secondo `SOURCES.md`.
9. DVS-Gesture e altri dataset confrontano soltanto baseline congelata e candidata finale.
10. La storia documentale rimane in Git. Non si creano snapshot, peer-review duplicate, task list
    parallele o copie del charter.
11. TCAP e PLIF mantengono seed 42, split, E0, augmentation e recipe. Non si modifica il gate in
    risposta al candidato. TCAP aggiunge soltanto proiezioni di feature passate, inizializzate a
    zero; PLIF aggiunge soltanto i parametri τ.
