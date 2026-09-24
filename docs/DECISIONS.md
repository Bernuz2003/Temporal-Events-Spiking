# Decisioni attive

**Aggiornate:** 2026-09-24

## Decisioni correnti della fase predittiva

1. F+DWC-3+TCAP-d8 è il riferimento congelato (C0). F+TCAP e MG+TCAP citati sotto sono riferimenti
   storici della discovery, non il modello corrente.
2. Prevale [`PREDICTIVE_TEMPORAL_PHASE1_AUDIT.md`](PREDICTIVE_TEMPORAL_PHASE1_AUDIT.md). La prima
   esecuzione non è evidenza sulle ipotesi; i suoi artifact sono conservati sotto
   `artifacts/superseded/`. L'audit A1–A4 e i probe R5 sono completi; l'esito è nella sua sezione 12.
3. **Regime per meccanismo.** L15 è una continuazione da C0, con BN a statistiche fisse, teacher in
   eval e controllo R0. D, S0 e S1 si addestrano **da zero** con la ricetta congelata di C0: in
   continuazione la rappresentazione non si muove (A3), e un router inserito a identità parte da un
   punto stazionario. I controlli dei bracci da zero sono i seed archiviati di C0 (42/43/44).
4. **Nessuna riproduzione di C0.** Il checkpoint congelato resta la base di confronto. L'equivalenza
   della ricetta è coperta da un test: lo scheduler corrente riproduce esattamente quello di C0 in
   tutte le 128 epoche. I bracci da zero partono dall'inizializzazione del backbone e dal flusso di
   dati della topologia C0 allo stesso seed.
5. **Nessun tetto in GPU-ore.** La disciplina sta nel disegno: ogni braccio risponde a una domanda,
   ha il proprio controllo, gira da solo sulla propria GPU. L'unità statistica di un'affermazione di
   superiorità è il seed, non l'epoca: seed 42 per primo, 43 e 44 per i bracci con firma coerente.
6. Nessun gate di +1 pp su seed singolo. Ogni decisione legge insieme il corredo di evidenza della
   sezione 9 dell'audit e viene registrata qui con il ragionamento.
7. **Verdetti.** P-F e P-0 chiusi (target fine non predicibile a nessun orizzonte). P-C sospeso.
   L15, D, S0 e S1 da eseguire nella forma corretta. Una sola fusione, strutturata
   (sorpresa → ampiezza, contenuto → allocazione), solo dopo due componenti positivi.
8. **Correzioni attive.** Ramp escluso dalla selezione e dalla finestra del gate di overfit;
   gradienti per blocco e regione su supporto identico; batch diagnostici stratificati per classe;
   autorità ausiliaria calibrata e rimisurata ogni epoca, con minimo verificato dal preflight;
   obiettivo S solo stage2 e solo regione attiva; routing solo stage2 con ampiezza separata
   dall'allocazione; sorpresa locale; rapporto costante fra learning rate discriminativi; L15 al
   solo prefisso di 1,5 s letto su denominatore fisso; gate dell'audit su ogni ingresso di training.
9. Si promuove un solo vincitore confermato, poi si riprendono le augmentation sulla candidata.
   B non riceve screen duplicati. I negativi validi restano parte dell'evidenza.
10. Official test escluso. Profiling dal checkpoint deployabile del proprio best; nessun risparmio
    attribuito automaticamente a gate soft. La disponibilità del teacher limita cosa può essere
    trasferito fra dataset.

## Decisioni della discovery al 13 settembre — contesto storico

Le seguenti voci documentano il percorso precedente. Le indicazioni di candidato corrente e
ordine futuro ai punti 6, 17, 21–26 sono superate dalle decisioni sopra; non autorizzano nuovi
run oltre al protocollo attivo. I risultati numerici conservano il proprio seed e significato.

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
6. **F+TCAP è il riferimento prestazionale corrente:** 52,79% accuracy e 52,36% F1, rispettivamente
   +7,98/+8,21 pp su B, con 492.516 parametri. Acc1/Acc2 sono 44,42/61,15%.
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
14. MultiGranular viene valutato con un solo encoder e un solo ramo configurabile. La prova di
    capacità usa fine input 320×32×32, downsampling spaziale appreso, mixing temporale MIMO 8:1 e
    fusione residua appresa: 480.036 parametri, ancora 20.672 sotto B. La configurazione Lite usa
    320×16×16, riduzione depthwise e somma diretta: 433.188 parametri. Un esito negativo di Lite da
    solo non può rigettare l'ipotesi multi-granular; Capacity misura l'utilità, Lite il limite Pareto.
15. Prima dei full, entrambe le configurazioni devono superare backward, gradienti finiti, causalità,
    allineamento prefix 8:1 e CUDA/AMP alla geometria DVS-Lip, oltre al bounded overfit.
16. E1 phase-count, EST, HATS, TORE/TAF, Matrix-LSTM e replica MSTP completa restano sospesi.
17. Quando emerge una candidata finale, baseline e candidata vengono replicate su due nuovi seed
    comuni. Si riportano media/deviazione, confronto appaiato, Acc1/Acc2, PrefixAUC e profilo del
    best. Solo allora si passa a augmentation e ottimizzazione.
18. Ogni conclusione hardware deve includere parametri, MAC multivalore, AC/SOP potenziali e ad
    attività, firing, stato/traffico e le due proxy Horowitz. Le proxy aritmetiche non sono joule
    misurati su FPGA e non includono memoria, routing, leakage o tutte le dinamiche LIF.
19. MG-Cap è positivo ma non Pareto: 48,42% F1 (+2,27 pp su F), 480.036 parametri, 673.792 elementi
    di stato e 20,30 h di training. Il guadagno è concentrato su Acc1 (+3,54 pp), mentre Acc2 sale
    di 0,60 pp. Riceve una sola combinazione con TCAP, senza ulteriore tuning interno.
20. I due clock-matched MG hanno fallito il gate. L'uguaglianza del decadimento fisico fra clock
    non è un'invarianza del percorso Conv-BN-LIF; la variante a gain 0,5 aumenta inoltre di circa
    sei volte il guadagno integrato. Il codice dedicato è stato rimosso tornando a `cde6ec3`.
21. Poiché MG+TCAP non ha raggiunto la soglia, prima del freeze resta un probe high-frequency
    locale. PMSN/PSN, GRU, LMU, Mamba, MTGA/graph e nuove rappresentazioni non ricevono full in
    questa fase perché richiedono un cambio di protocollo o più run di attribuzione.
22. F+MG-Cap+TCAP è il record development single-seed: 53,52% accuracy e 53,15% F1, +8,71/+9,00
    pp su B. MG aggiunge però soltanto 0,73/0,80 pp a F+TCAP, con IC95% appaiato F1
    `[−1,16; +2,71]` e McNemar `p=0,479`; non raggiunge la soglia preregistrata di +2 pp.
23. **F+TCAP resta il finalista strutturale primario.** MG+TCAP viene conservato come record
    esplorativo e candidato di latenza, perché migliora il F1 a 1 s di 5,05 pp e il F1-PrefixAUC
    di 2,34 pp, ma costa +34,16% stato, +23,08% MAC, +25,89% SOP, +22,45% Horowitz ad attività e
    circa 2,9 volte il tempo di training rispetto a F+TCAP.
24. TCAP è il risultato architetturale robusto: aggiunge +6,21 pp F1 a F e +4,73 pp a MG. Prima di
    modificare i ritardi si esegue un'ablation checkpoint-only dei tap 1/2/4. Un solo run
    `[1,2,4,8]` è ammesso se rimuovere `d=4` costa almeno 1 pp e il suo contributo non è inferiore
    a `d=2`; non si esegue uno sweep dei ritardi.
25. La discovery può ricevere un solo probe high-frequency: F+TCAP con mixer locale depthwise 3×3
    nel primo stage, E0 e stage 2 invariati. La promozione richiede +2 pp F1 oppure parità entro
    0,5 pp con vantaggio hardware misurato. Dopo il probe e la diagnostica condizionale dei tap
    l'architettura viene
    congelata.
26. La fase successiva separa conferma multi-seed, trasferimento DVS-Gesture, raffinamento
    supervisionato, SSL/predictive pretraining, compressione e infine test ufficiale. La sequenza e
    le stop rule sono in `VALIDATION_REFINEMENT_ROADMAP.md`.
27. PLIF non viene reinserito come neuron model. La sua persistenza post-evento e il vantaggio ai
    prefissi motivano una loss ausiliaria ai prefissi tardivi e, più avanti, un eventuale arresto
    adattivo; entrambe sono valutate sulla struttura congelata.
