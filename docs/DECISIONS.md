# Decisioni attive

**Aggiornate:** 2026-09-09

1. DVS-Lip resta il benchmark di sviluppo e l'official test resta embargoed fino alla valutazione
   finale. Tutte le cifre correnti sono seed 42 sulla development validation ricavata
   dall'official-train.
2. La baseline B è MiniQKFormer E0 da 500.708 parametri, mean readout fixed-window, 40×50 ms:
   accuracy 44,81%, Macro-F1 44,15%. I modelli 1M/2M sono controlli di capacità, non candidati.
3. La selezione architetturale conserva optimizer, schedule, clipping, augmentation e loss. Il
   tuning e l'augmented training iniziano soltanto dopo il freeze della struttura.
4. F è promosso come front-end Pareto: 46,15% F1 (+1,99 pp), −13,91% parametri, −56,61% stato,
   −47,64% SOP potenziali e −30,60% Horowitz densa. Il firing maggiore non annulla la riduzione
   strutturale, ma va sempre riportato.
5. TCAP è promosso come prova positiva del temporal core: 48,12% F1 (+3,97 pp), con +12,27%
   parametri, +16,44% MAC multivalore e +14,66% Horowitz attività. Il segnale giustifica il
   trasferimento su F; la compressione depthwise T è rinviata alla fase Pareto post-freeze.
6. Il prossimo candidato prestazionale è **F+TCAP**. La composizione ha 492.516 parametri e un costo
   strutturale previsto inferiore a B; attività ed energia saranno accettate soltanto dal profilo
   del proprio best. Complementarità degli errori F/TCAP rende il run informativo, senza garantire
   additività dei guadagni.
7. **B+T non viene eseguito ora.** T è il FIR depthwise da 576 coefficienti che comprime TCAP
   eliminando il mixing cross-channel. È un esperimento di ottimizzazione, non una candidata con
   maggiore potenziale prestazionale, e sarà rivalutato soltanto sull'architettura congelata.
8. PLIF non è promosso per il punteggio finale: 43,68% F1 (−0,47 pp). La diagnostica completa gli
   assegna però +3,47 pp di F1 PrefixAUC rispetto a B e circa +10 pp F1 tra `L+100` e `L+300 ms`.
   Mantiene attività semantica profonda per 300–500 ms su input nullo, coerentemente coi τ appresi.
9. Il denominatore `sum/t` contro `sum/40` cambia poco le curve. Nessun cutoff event-aligned PLIF
   supera il F1 finale di B; `last_event+K` è quindi respinto. PLIF resta un risultato di latenza e
   dinamica. Supervisione ai prefissi o arresto adattivo sono eventuali lavori post-freeze.
10. Gated-v2 ha fallito il bounded overfit e il full manuale non ha mostrato recupero entro 28
    epoche. L'esatta forma diagonale/tanh è chiusa. Il fallimento storico resta invalido per via
    dell'inizializzazione, ma quello corretto è evidenza sufficiente per non spendere altri run.
11. F+TBR ha fallito il gate esclusivamente sulla loss: accuracy train 100% e validation 98,44%,
    loss validation minima 1,5368. La codifica permette memorizzazione ma produce separazione dei
    logit insufficiente sotto la ricetta invariata. Il full resta bloccato e non si provano bit,
    `Δt`, polarità o normalizzazioni alternative.
12. Il TBR canonico pubblicato scarta polarità e molteplicità intra-micro-bin; il confronto locale
    usa quindi un canale ed è fedele a questa semantica. Spike-TBR usa `β=0,9` e soglia `1,1`, ma
    viene chiamato paper-aligned e non replica ufficiale: non è disponibile codice sorgente e il
    paper non determina completamente `w(p)` né la continuità della membrana fra finestre `ΔT`.
13. Spike-TBR ha superato il gate, ma nello snapshot epoca 38 è a 8,76% F1. L'encoder emette solo
    il 9,82% dei voxel macro non nulli di TBR sui 64 sample diagnostici; il reset ogni 50 ms spezza
    accumuli sub-soglia. Il full può terminare per completezza, ma non si lancia una variante di
    reset: il paper non specifica la continuità e un secondo run sarebbe tuning locale.
14. MultiGranular-Lite è ora definito e implementato come singolo candidato: E0 full-spatial a 40
    step più count ON/OFF a 320 step e 16×16, ramo spiking `2→16→64`, riduzione temporale causale
    depthwise 8:1 e fusione additiva prima dello stage 1. Il backbone resta a 40 step, l'input cresce
    del 12,5%, non c'è endpoint oracle e non sono aperti sweep. È autorizzato al bounded overfit.
15. E1 phase-count, EST, HATS, TORE/TAF, Matrix-LSTM e replica MSTP completa restano sospesi.
16. Le quattro macchine sono capacità massima, non un obbligo a riempire una griglia. Una macchina
    libera può eseguire il gate MultiGranular-Lite; la seconda resta libera finché F+TCAP non
    conclude. Non si crea un'altra rappresentazione per saturarla.
17. Quando emerge una candidata finale, baseline e candidata vengono replicate su due nuovi seed
    comuni. Si riportano media/deviazione, confronto appaiato, Acc1/Acc2, PrefixAUC e profilo del
    best. Solo allora si passa a augmentation e ottimizzazione.
18. Ogni conclusione hardware deve includere parametri, MAC multivalore, AC/SOP potenziali e ad
    attività, firing, stato/traffico e le due proxy Horowitz. Le proxy aritmetiche non sono joule
    misurati su FPGA e non includono memoria, routing, leakage o tutte le dinamiche LIF.
