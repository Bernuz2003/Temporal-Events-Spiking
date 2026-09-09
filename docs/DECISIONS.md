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
8. PLIF non è promosso per il punteggio finale: 43,68% F1 (−0,47 pp). Non si provano altri subset di
   layer, inizializzazioni o combinazioni durante discovery. La sua elevata PrefixAUC e i τ appresi
   autorizzano soltanto la diagnostica checkpoint-only B/PLIF, non `last_event+K` né un nuovo full.
9. La diagnostica temporale separa `sum(h[1:t])/t`, `sum(h[1:t])/40` e attività post-evento; misura
   margini, stabilità e firing per layer. L'allineamento all'ultimo evento è oracle e viene usato
   solo per spiegazione meccanicistica. Un eventuale uso futuro richiederà supervisione ai prefissi
   o arresto adattivo validato, non una costante scelta sul validation set.
10. Gated-v2 ha fallito il bounded overfit e il full manuale non ha mostrato recupero entro 28
    epoche. L'esatta forma diagonale/tanh è chiusa. Il fallimento storico resta invalido per via
    dell'inizializzazione, ma quello corretto è evidenza sufficiente per non spendere altri run.
11. La rappresentazione E0 a 40 bin è una variabile strutturale esplicita. I probe prioritari sono
    **F+TBR** e **F+Spike-TBR-LIF**: 8 micro-bin da 6,25 ms sono compressi in ciascuno dei 40
    macro-bin da 50 ms. Nessuno sweep di bit, `Δt`, `β` o soglia viene aperto.
12. Il TBR canonico pubblicato scarta polarità e molteplicità intra-micro-bin; il confronto locale
    usa quindi un canale ed è fedele a questa semantica. Spike-TBR usa `β=0,9` e soglia `1,1`, ma
    viene chiamato paper-aligned e non replica ufficiale: non è disponibile codice sorgente e il
    paper non determina completamente `w(p)` né la continuità della membrana fra finestre `ΔT`.
13. E1 phase-count resta implementato ma sospeso. MultiGranular-Lite è una candidata successiva
    ad alto potenziale, non ancora un run: prima vanno fissati ramo fine, fusione e costo. EST,
    HATS, TORE/TAF, Matrix-LSTM e replica MSTP completa restano fuori dalla campagna corrente.
14. Le quattro macchine sono capacità massima, non un obbligo a riempire una griglia. L'iterazione
    usa tre full indipendenti e una diagnostica: F+TCAP, F+TBR, F+Spike-TBR-LIF e B/PLIF
    checkpoint-only. Nessuna combinazione parte prima dei risultati.
15. Quando emerge una candidata finale, baseline e candidata vengono replicate su due nuovi seed
    comuni. Si riportano media/deviazione, confronto appaiato, Acc1/Acc2, PrefixAUC e profilo del
    best. Solo allora si passa a augmentation e ottimizzazione.
16. Ogni conclusione hardware deve includere parametri, MAC multivalore, AC/SOP potenziali e ad
    attività, firing, stato/traffico e le due proxy Horowitz. Le proxy aritmetiche non sono joule
    misurati su FPGA e non includono memoria, routing, leakage o tutte le dinamiche LIF.
