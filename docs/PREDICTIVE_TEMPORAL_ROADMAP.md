# Roadmap operativa: supervisione predittiva e memoria temporale condizionale

**Definita:** 2026-09-19. **Stato:** protocollo e infrastruttura implementati nel branch
`Predictive-Temporal-Coding`; test CPU superati, verifica CUDA/AMP SMILIES ancora necessaria.
Questo documento non attesta l'esecuzione di nuovi esperimenti.

La fase precede la ripresa delle augmentation. Mantiene come riferimento **F+DWC-3+TCAP-d8** e
riapre soltanto le ipotesi descritte qui. Motivazioni, anteriorità e limiti teorici sono nella
[review scientifica](PREDICTIVE_TEMPORAL_RESEARCH_REVIEW.md). Non si riaprono ricerca sui neuroni,
binning, dimensioni del backbone o grandi teacher.

## 1. Obiettivo, riferimento e risultati da non confondere

L'obiettivo primario è migliorare il **Macro-F1 finale a 2 s** rispetto alla stessa architettura
con lo stesso budget di training. Il secondo obiettivo è anticipare decisioni utili senza perdere
qualità finale. La riduzione energetica è una conclusione separata, da misurare.

| Riferimento DVS-Lip | Macro-F1 % | Ruolo |
|---|---:|---|
| Congelata seed 42 | 55,18 | checkpoint iniziale C0 per lo screening |
| Congelata seed 43 | 54,88 | riferimento della replica 43 |
| Congelata seed 44 | 56,56 | riferimento della replica 44 |
| Congelata, media ± SD campionaria | 55,54 ± 0,90 | variabilità osservata, non soglia scelta sul seed migliore |
| Spatial erasing seed 42 | 56,69 | risultato augmentation separato, da conservare e riesaminare dopo questa fase |

I valori sono development validation. Il test ufficiale resta escluso, anche per teacher,
normalizzazione, diagnostiche e profiling. Un nuovo seed 42 non va confrontato selettivamente
con il migliore fra 42/43/44. B non viene riaddestrata: il controllo scientifico è ora la candidata
congelata, con lo stesso ulteriore training della variante.

Per C0 si usa `dvslip_f_tcap_stage1_dwc3_d8__20260914_093427_434930__seed42/best.pt`, con hash
registrato. I checkpoint MG e PLIF rimangono risorse diagnostiche, non vincitori presunti.

## 2. Domande scientifiche e criteri di risposta

| ID | Domanda | Confronto che può rispondere |
|---|---|---|
| Q1 | Predire un target futuro fine migliora la rappresentazione deployabile? | CRP contro controllo con uguale continuazione |
| Q1a | Serve il futuro, oppure basta distillare il presente fine? | CRP contro distillazione fine simultanea |
| Q1b | Serve la risoluzione fine, oppure basta predire feature coarse? | CRP contro predizione futura coarse |
| Q2 | Il contenuto corrente aiuta a scegliere il peso dei ritardi? | TCAP dinamico contro statico; diagnostica gate dinamici contro gate costanti |
| Q3 | L'errore predittivo aggiunge informazione al controllo della memoria? | Stessa auxiliary loss/predittore, con e senza errore disponibile al router |
| Q4 | I benefici di supervisione e routing si sommano? | Una sola fusione contro entrambi i componenti già positivi |
| QL | Possiamo anticipare la decisione senza introdurre PLIF? | Curve ai prefissi per ogni candidato; eventuale distillazione tardiva come fallback |

Le domande sono distinte. Un aumento di F1 non autorizza da solo una spiegazione meccanicistica,
e un aumento di PrefixAUC non equivale a un aumento di accuracy finale.

## 3. Scelta di budget: continuazione controllata, non nuovi full da 128 epoche

La discovery usa **64 epoche aggiuntive** dal medesimo C0, senza modificare la durata per
salvare un candidato. Ogni braccio parte nuovamente da C0, mai dal best di un altro braccio,
incluse le fusioni. Così una combinazione non riceve più aggiornamenti dei singoli componenti.

La capacità aggiuntiva viene assegnata al meccanismo studiato prima della compressione:
predictor cross-resolution spaziale training-only, router TCAP locale e predictor surprise
MIMO. Backbone, classificatore, embedding e matrici TCAP restano invariati. Le formulazioni
lineari, globali e depthwise restano ablation future di compressione, non l'unico test con cui
rigettare l'ipotesi.

Questo è uno studio di fine-tuning dopo pretraining supervisionato comune. Un risultato negativo
chiude questa formulazione nel budget assegnato; non dimostra che il metodo non possa funzionare
da zero o con un lungo pretraining. Un risultato positivo non viene presentato come confronto
fra architetture addestrate tutte da zero.

| Elemento | Scelta fissata per tutti i bracci |
|---|---|
| Recipe di fase | `dvslip_predictive_continuation_64_v1`, implementata e vincolata dal workflow dedicato |
| Dati e input deployato | split development esistente; E0, 40×50 ms; mean/fixed-window |
| Augmentation | solo flip orizzontale 0,5 già presente in C0; niente nuove augmentation |
| Ottimizzazione | AdamW nuovo, senza momenti ereditati; batch 16, accumulo 2 |
| Schedule | 64 epoche; LR massimo `1e-4`, minimo `1e-6`, cosine; warmup 2 epoche da fattore 0,01 |
| Altri parametri | weight decay `5e-4`, label smoothing 0,1, clipping 1,0, AMP |
| BatchNorm student | running mean/variance di C0 fisse in tutti i bracci; affine apprendibile |
| Teacher | pesi e statistiche fissi; modalità eval; nessuna augmentation indipendente |
| Selezione | massimo Macro-F1 validation a 2 s; stesso numero di occasioni di selezione |
| Randomness | seed 42; ordine dati e trasformazioni riproducibili con stream RNG separato dai nuovi moduli |

Le scelte numeriche sono convenzioni pragmatiche preregistrate, non ottimi garantiti dalla
letteratura. Il LR è ridotto rispetto alla discovery perché si parte da un modello allenato.
Non si trasferisce questo LR come tuning universale ad altri dataset.

Non si prolunga post-hoc un candidato negativo. Un'estensione comune a 96/128 epoche può essere
preregistrata solo dopo lo screen per un candidato già positivo la cui curva a epoca 64 sia ancora
crescente, insieme al controllo R0 con identico orizzonte. Non fa parte degli otto screening.

**R0, controllo obbligatorio:** stessa continuazione senza auxiliary loss né router.
L'epoca 0 viene valutata e registrata separatamente: deve riprodurre C0 entro tolleranza numerica.
Non viene usata per nascondere il fallimento della continuazione. Si riportano best post-update,
ultima epoca e C0. Se il best post-update perde oltre 0,5 pp F1 da C0, la campagna si ferma per
discutere la ricetta comune; non si aggiustano learning rate diversi per candidato.

Il controllo R0 precede gli screening lunghi. La scelta di congelare le statistiche BN impedisce
che il contesto student utilizzi statistiche dei timestep futuri e rende omogenea la comparazione.
Non va confusa con `eval()` dell'intero student, che disabiliterebbe altri comportamenti di training.

## 4. P0 — diagnostiche e contratto di causalità prima del training

### P0.1 Integrità e costo

Verificare checkpoint, split, allineamento coarse/fine, target e shape; reset degli stati per
sequenza; nessuna dipendenza dal futuro nei percorsi dichiarati causali, anche in train con BN
fissa. Perturbare eventi successivi a t non deve cambiare il contesto fino a t. Il teacher può
vedere il target futuro, ma quel target non deve entrare nel forward student o nel router.

Prima di allocare training: breve misura di throughput/VRAM e stima del costo di teacher,
predittore e gate. La shape uguale non prova che due feature siano semanticamente allineate.
I workflow generici `candidate`/`refine` non implementano questo contratto. Il comando dedicato
`predictive-continuation` esegue preflight, gate warm-start, continuazione da C0 ed export
deployabile; non deve essere sostituito con un training fresco accidentale.

### P0.2 Target predittivo: controllo di fattibilità, non selezione su validation

Il teacher fine iniziale è il ramo MG del checkpoint
`dvslip_f_multigranular_temporal_capacity__20260912_135802_779469__seed42/best.pt`, prima
dell'ultimo LIF. È supervisionato e co-adattato alla fusione: non lo definiamo teacher ottimale.
Il contesto student è l'uscita stage1, comprendente la storia causale già codificata.

Si fissa **h=2 macro-bin, 100 ms**: è un anticipo moderato, non scelto per massimizzare la
predicibilità sulla validation e non derivato da un presunto tau universale. I tap `[1,2,4,8]`
restano invariati; l'orizzonte della loss non è il raggio della memoria.

Un probe lineare sui checkpoint congelati confronta target presente e futuro, media di training
e persistenza. Fit e holdout diagnostico sono ricavati esclusivamente dal development-train,
con partizione per sample, mai per finestre sovrapposte. Stessa partizione per tutti i probe;
nessun dato holdout entra in fit o normalizzazione. Questo holdout serve al probe, non è una
nuova validation per selezionare decine di varianti; dopo il probe i training usano l'intero train.

Si riportano errore normalizzato, varianza dei target e skill rispetto ai riferimenti banali,
separati per attività e coda. Il probe è un lower bound affine `1×1`, mentre il predictor
discovery può compensare movimento locale: un risultato debole non falsifica il principio.
Target degenerati, leakage o allineamento errato fermano P-F; skill affine positiva lo rafforza,
ma non è più un veto contro l'unica prova preregistrata. Un esito positivo resta solo fattibilità.

### P0.3 Errore predittivo e valore della storia

Agli ingressi TCAP, confrontare il predittore causale convesso depthwise con persistenza e media
dei tap come baseline diagnostica compressa.
Usare errori normalizzati per scala dei canali, stimata sul train; valutare anche attività,
transizioni e coda. Il confronto deve restare favorevole anche fuori dalla sola coda.

Verificare se l'errore aggiunge informazione rispetto ad ampiezza/event rate sul danno della
rimozione della storia, valutato sui sample di holdout. Il danno è diagnostico: l'ablazione di
una rete co-adattata non è il controfattuale di una rete riaddestrata. Le etichette di training
possono servire a questa analisi, mai al router a inference.

Il suo fallimento non rigetta il predictor MIMO/spaziale usato da S: segnala quanto costa la
formulazione compressa. S0 è il controllo addestrato che stabilisce se il predictor capace
apprende un residuo utile. Target degenerati o leakage fermano comunque il ramo.

## 5. P — supervisione predittiva, primo asse

**P-F:** CE finale più predizione del target fine a t+2. Il predictor training-only condiviso
nel tempo usa `DWConv 3×3 → Conv 64→128 → GELU → Conv 128→64` (17.152 parametri), così può
modellare movimento locale senza aumentare il modello deployato. Non si tenta di
ricostruire eventi singoli o spike binari. Target e predizione sono confrontati nello spazio
pre-LIF standardizzato per canale con statistiche del training e scala minima dichiarata.

Si usa SmoothL1 mediata su feature/posizioni e tempi, con peso massimo **0,1**, portato da zero
a tale valore nelle prime quattro epoche. Il termine CE mantiene peso uno. La quantità di target
non deve moltiplicare implicitamente la forza della loss. Questa scelta viene mantenuta anche
nei controlli; si registrano separatamente loss e norme dei gradienti dei due obiettivi.

Per la loss predittiva, ogni sample pesa ugualmente; si usano le coppie con target entro l'ultimo
bin raw occupato. Gli eventuali bin vuoti interni restano validi. Si esclude così la lunga coda
artificiale dall'obiettivo, senza affermare che le sue feature siano nulle. L'endpoint è metadato
offline della loss, non input del modello. Tutti i sample, incluse eventuali sequenze prive di
coppie valide, conservano la CE finale. Si registra la copertura dei target.

Per comparare presente e futuro senza confondere durata e target, entrambi i controlli usano
gli stessi indici di contesto validi per P-F, lo stesso numero di coppie e la stessa normalizzazione
della loss. Il teacher deve ricevere la stessa trasformazione spaziale dello student; un flip
delle feature cached non viene assunto equivalente a ricodificare l'input con un encoder appreso.

**Controlli condizionati al successo di P-F:**

| ID | Unica domanda aggiunta | Target |
|---|---|---|
| P-0 | Serve anticipare? | stesso teacher fine a t |
| P-C | Serve l'informazione fine? | feature stage1 del teacher C0 coarse a t+2 |

P-C mantiene la stessa geometria del target e la stessa classe/capacità di predictor. Non usa
un teacher più grande. Se un controllo eguaglia P-F entro 0,5 pp, non rivendichiamo superiorità
del meccanismo più complesso; preferiamo il più economico, tenendo conto della variabilità.
Se P-0 è migliore, resta una distillazione utile, ma la tesi non attribuisce il guadagno al futuro.

## 6. D — TCAP dinamico, secondo asse indipendente

**D:** mantenere tap e matrici TCAP; introdurre pesi per tap dipendenti dal contenuto corrente
alla risoluzione della feature TCAP. Nessun delay apprendibile.

\[
y_t=x_t+\sum_d g_{t,d,h,w}W_dx_{t-d},\qquad
g_{t,d,h,w}=2\sigma(R(x_{t,:,h,w})_d).
\]

`R` è un MLP locale `C→C/2→4` implementato con convoluzioni `1×1`. L'ultimo affine
inizia a zero: sul checkpoint allenato la funzione iniziale coincide con C0, con gate unitari.
I due router aggiungono 10.728 parametri. La decisione locale evita che GAP diluisca una
transizione confinata alla bocca; operazioni e traffico vengono profilati. P-F e D possono
procedere in parallelo dopo R0: entrambi partono da C0.

Sul best di D, confrontare senza riaddestrare gate dinamici e gate costanti pari alla media
stimata sul train. Registrare distribuzioni dei gate e contributi effettivi `g*W*x`, non soltanto
pesi o norme. Se la sostituzione non cambia le prestazioni, il risultato sostiene ricalibrazione
o regolarizzazione, non dimostra routing utile. Anche un calo per gate costanti è evidenza
diagnostica, non sostituisce una futura ablation riaddestrata qualora si volesse una rivendicazione
forte di novità architetturale.

## 7. S — surprise come informazione aggiuntiva, senza imporne il segno

Questo asse si apre solo dopo P0.3 e l'esito di D. Il residuo non viene interpretato automaticamente
come «passato sbagliato»; non imponiamo che alta sorpresa sopprima la memoria.

Si confrontano **S0 e S1**, entrambi da C0 con lo stesso predittore, la stessa loss predittiva e
lo stesso budget. Per ogni tap il predictor applica una depthwise `3×3` causale e una proiezione
MIMO `C→C`; somma poi i quattro contributi. Aggiunge 88.832 parametri e può modellare movimento
locale e dinamiche cross-channel. I kernel spaziali partono come identità e le proiezioni come
media diagonale dei tap. La loss ricostruisce x corrente con target stop-gradient;
scala, peso 0,1 e ramp-up sono fissati come nel ramo P. Non si sommano ancora le loss P e S.

| Braccio | Accesso del router all'errore | Ruolo |
|---|---|---|
| S0 | nessuno | controllo del beneficio della sola auxiliary loss/predittore |
| S1 | residuo normalizzato per canale `C→4` | misura il valore dell'errore per il routing |

Se D ha superato lo screen, entrambi contengono il router di contenuto D; altrimenti S1 usa solo
coefficienti per tap moltiplicati per l'errore, senza bias né termine di contenuto.
Questa diramazione è decisa dall'esito di D, non scegliendo dopo quale variante S funzioni meglio.
In S1 la proiezione lineare dell'errore per canale aggiunge 768 parametri; parte da zero e può
apprendere entrambi i segni.
Non si sovrappongono due gate moltiplicativi con scale non identificabili.

Il residuo passato al router è staccato dal gradiente: la classificazione allena i coefficienti
del router, non deforma direttamente il predittore per fabbricare il segnale. L'auxiliary loss
allena predittore e storia student. Il target resta mobile perché le feature student cambiano:
si controllano varianza, skill predittiva e contributi CE/auxiliary durante tutto il training.

Per S la loss usa tutta la finestra, con media prima per sample e poi per due regioni: fino
all'ultimo evento e coda, peso uguale alle regioni presenti. Non si introduce `last_event+8`.
Le metriche delle due regioni restano separate; a inference non viene passato alcun endpoint.
Le statistiche del router sono causali, senza normalizzazione sulla sequenza intera.

Per promuovere S1 servono vantaggio sul controllo S0 e assenza di collasso del predittore. Se
migliorano entrambi allo stesso modo, il beneficio appartiene alla supervisione ausiliaria.
Se S0 è migliore, si conserva eventualmente quella ricetta; si chiude la proposta di surprise routing.
Non si dichiara riduzione dei MAC grazie a gate soft che non saltano materialmente operazioni.

## 8. Come sfruttiamo PLIF, senza riaprire il neuron model

La diagnostica B/PLIF ha mostrato **+3,47 pp di F1-PrefixAUC** e circa **+10 pp F1 fra L+100 e
L+300 ms**, con risultato finale invece inferiore a B di 0,47 pp. Sono riferimenti di quel
confronto, non vantaggi già osservati sul finalista attuale.

Da questo derivano tre scelte concrete:

1. Per tutti i best, valutare la curva a prefissi ogni 50 ms e confrontarla con R0; osservare
   separatamente miglioramento prima della fine dell'input e durante la coda.
2. Valutare diagnostiche event-aligned con endpoint oracle e quota di clipping esplicite, senza
   utilizzarle per definire un readout deployabile o scegliere una costante di settling.
3. Tenere un **fallback L** di supervisione dei prefissi se P-F viene fermato dal probe o non
   supera lo screen. L sostituisce i controlli P-0/P-C, non si aggiunge a una ricerca P già positiva.

**L:** teacher C0 congelato a finestra completa; student supervisionato con CE finale e
distillazione soft dei logit ai prefissi fissi 1,0 e 1,5 s. KL teacher→student, temperatura 2,
fattore T² convenzionale, media dei due prefissi e peso massimo 0,1 con ramp-up di quattro epoche.
Non si impone una label dura ai primi 500 ms. Un target soft non elimina l'ambiguità dei prefissi,
ma evita di trattare ogni prefisso come parola già completamente osservata.

È una variante di supervisione tardiva motivata da PLIF, non una replica di MEOM né un trasferimento
automatico della sua dinamica. Il teacher è il finalista più accurato; PLIF non viene scelto come
teacher globale meno accurato senza evidenza di complementarità sul modello corrente.
Il beneficio di L può riguardare solo latenza: in quel caso viene riportato come tale, senza
spacciarlo per soluzione al punteggio finale. L può occupare il ruolo di supervisione nella
fusione solo se supera anche il criterio di accuracy principale.

## 9. Criteri preregistrati: avanzare, fermare, replicare

Le soglie sono regole decisionali pratiche, non test di significatività né valori estratti dai
nuovi risultati. Per P-F, D, S0 e L il controllo comune è R0. Per attribuire il routing S1 il
controllo è S0, oltre al confronto con R0.

| Esito seed 42 | Decisione |
|---|---|
| ≥+1,0 pp F1 sul controllo, accuracy non inferiore di oltre 0,5 pp | segnale prestazionale materiale, ammesso ai controlli/alla selezione |
| ≥+2,0 pp F1 | priorità alta per la conferma multi-seed |
| ΔF1 inferiore a +1,0 pp | nessuna fusione prestazionale o tuning per inseguire il margine |
| F1 finale entro −0,5 pp, F1-PrefixAUC normalizzata ≥+2,0 pp | candidato di latenza distinto; non sostituisce il vincitore di accuracy |
| NaN, target degenerati, gradienti assenti o causalità violata | run non valido; correggere il difetto, non classificarlo come evidenza negativa dell'idea |

Per S1 il gate di +1 pp deve essere soddisfatto rispetto a S0 e il risultato deve superare R0
di almeno +1 pp. Il valore della predizione non è dimostrato se il router cresce ma il predittore
non batte più i riferimenti banali.

Si riportano intervalli bootstrap appaiati stratificati per classe per i delta di validation.
Un IC che attraversa zero resta un esito incerto, anche se supera la soglia pratica: può motivare
la replica, non una dichiarazione di superiorità. La selezione sullo stesso validation e fra
più candidati resta esplorativa; il bootstrap entro seed non sostituisce la variabilità fra seed.

Un componente prima di entrare in una fusione deve superare lo screen e avere un'interpretazione
compatibile con i controlli. Nessuna soglia o budget viene abbassata dopo un risultato deludente.

## 10. Una sola fusione, poi conferma e trasferimento

Si sceglie un vincitore della supervisione (P-F, P-0, P-C, oppure L) e uno del routing (D oppure
S1, che può già contenere D). Se solo uno è positivo, non si forza la combinazione.

**FUS:** i due vincitori insieme, da C0 per le stesse 64 epoche. Per conservare la forza delle
singole loss, i pesi già fissati non cambiano; qualora siano presenti due auxiliary loss si
registrano separatamente anche le norme del gradiente totale. L'aumento dell'obiettivo ausiliario
è un'interazione da dichiarare, non prova automatica di sinergia. Non si aggiunge una griglia di
pesi per recuperare un'eventuale interferenza.

La fusione deve migliorare il migliore componente di almeno **0,5 pp F1**, conservando il
vantaggio materiale su R0. Se non passa, si seleziona il migliore singolo. Questa sola fusione
può già riunire tutte e tre le idee: supervisione cross-resolution + router di contenuto + errore.

Solo il vincitore finale e R0 proseguono ai seed **43 e 44**, ciascuno dal proprio C0 già
disponibile. Teacher MG seed 42 e protocollo target restano fissi: la replica misura variabilità
dello student, non quella della scelta del teacher. Eventuale sensitivity del teacher resta
un limite dichiarato, non un nuovo sweep.

Promozione finale: delta F1 medio appaiato ≥+1 pp rispetto a R0, positivo in almeno due seed su
tre, nessun seed sotto −0,5 pp; riportare comunque tutti i valori, SD, intervalli entro seed,
accuracy e profilo. Con tre seed non si afferma una certezza statistica generale. Se la regola
non passa si mantiene C0 o, se migliore e confermato, il solo controllo di continuazione.

Per un metodo promosso segue DVS-Gesture seed 42: stessa struttura/direzione della loss e
controllo di continuazione appaiato, usando il checkpoint Gesture congelato. Parametri del
protocollo dataset-specific rimangono quelli Gesture, senza flip e senza tuning dei tap.
L'orizzonte h=2 resta in bin; non si sostiene identità del tempo fisico fra dataset.

Il trasferimento P-F/P-0 richiede un teacher fine Gesture comparabile, che oggi non è attestato:
non si usa MG-Lip come teacher Gesture senza uno studio distinto. Se manca, si trasferisce solo
la componente implementabile (routing, oppure futura supervisione coarse già validata) e si
dichiara il trasferimento **parziale**. Un nuovo teacher MG-Gesture è fuori dal budget corrente.
Non si promette quindi generalizzazione dell'intera pipeline prima di poterla valutare.

## 11. Sequenza operativa, parallelismo e tetto di spesa

| Ordine | Attività | Nuovi training di continuazione | Dipendenza |
|---:|---|---:|---|
| 0 | P0: causalità, target, probe, diagnostica PLIF-informed, throughput | 0 | checkpoint e contratto verificati |
| 1 | R0, controllo comune | 1 | P0 integrità |
| 2 | P-F e D in parallelo | fino a 2 | R0 valido; P-F richiede P0.2 |
| 3 | P-0 e P-C **oppure** fallback L | fino a 2 **oppure 1** | esito di P-F |
| 4 | S0 e S1 in parallelo | fino a 2 | P0.3 valido e risultato di D |
| 5 | unica fusione | fino a 1 | componenti positivi e controllati |
| 6 | vincitore + R0, seed 43/44 | 4 | candidato seed 42 scelto |
| 7 | trasferimento possibile + controllo Gesture seed 42 | fino a 2 | conferma Lip, teacher disponibili |

**Massimo screening: 8 training da 64 epoche**, non otto full da 128; spesso meno perché i rami
sono condizionali. Massimo fino a conferma Lip: 12 continuazioni; con trasferimento: 14.
Otto continuazioni hanno 512 epoche student totali, nominalmente quattro full storici; non sono
equivalenti in ore GPU perché il teacher può aggiungere costo significativo.

Prima del primo run P si converte il throughput misurato in un preventivo e si registra un tetto
in GPU-ore. Il budget discovery pianificato è **cinque volte** il training del full C0 seed 42;
il limite invalicabile è **sei volte**, comprensivo di controllo, teacher online, probe, gate e
fusioni. Conferma e trasferimento hanno budget separato. Con 8,972 ore per C0, i due valori sono
**44,86** e **53,83 GPU-ore**;
il budget dei nuovi run include anche le valutazioni necessarie, non soltanto gli optimizer step.
Non basta restare entro il numero di run. Se il preventivo non rientra, si sospende
per primo S, poi la fusione; non si tagliano i controlli necessari per attribuire P o D.
Il confronto usa il tempo C0 del server di riferimento e registra hardware e throughput per
evitare di interpretare differenze fra server come costo intrinseco del metodo.

Non occorre tenere tutte le risorse occupate: parallelismo solo fra confronti già determinati
da informazioni disponibili. I run in corso della fase augmentation possono terminare e restano
nel registro; nessuno viene interrotto o cancellato da questo piano.

## 12. Requisiti prima dei lanci e output obbligatori

Prima di avviare un braccio: test di forma, backward e gradienti utili nei nuovi moduli;
causalità train/eval; CUDA/AMP; invarianza iniziale rispetto a C0 dove promessa; conservazione
di reset, allineamento e trasformazioni teacher/student; subset overfit previsto dalla recipe.
Il gate deve verificare la CE separatamente dalla loss totale: la soglia 1,5 non viene applicata
alla somma arbitraria di CE e ausiliarie. Per warm-start si usa una copia del checkpoint e si
controlla che il successo non mascheri un predictor/router privo di gradienti. I pesi modificati
nel gate non entrano nella continuazione di produzione. Non si allenta il gate dopo l'esito.

Il manifest di ogni run conserva: hash C0/teacher; parent checkpoint; recipe nuova; seed dei dati
e moduli; stato BN; maschere/pesi delle loss; normalizzazione fittata solo sul train; epoche,
aggiornamenti e GPU-ore; configurazione/commit/ambiente; best, last e predizioni per sample.

| Categoria | Output necessario |
|---|---|
| Qualità finale | Macro-F1, accuracy, Acc1/Acc2, confusioni e delta appaiati contro R0 e C0 |
| Dinamica | F1/accuracy ogni 50 ms sul best, PrefixAUC normalizzata su intervallo fisso comune; distinguere dalla AUC sui soli prefissi configurati |
| Event-aligned | endpoint oracle dichiarato, attività/coda e quota clippata; nessun cutoff selezionato |
| Training | CE e ogni auxiliary loss separate; gradienti, clipping/overflow, best/last, tempo, VRAM |
| Teacher/predizione | varianza dei target, skill rispetto a persistenza/media, copertura, stabilità train/holdout |
| Routing | gate e contributi per tap, controllo a gate costanti, residuo normalizzato; niente inferenze da sole norme |
| Hardware | parametri deployati e training-only separati, stato/traffico, MAC multivalore, AC/SOP potenziali e ad attività, firing rate, Horowitz densa/ad attività |

Il profilo viene dal best del candidato; se manca non si rivendica superiorità Pareto. Nessun
gate soft viene contabilizzato come skip automatico; nessuna proxy aritmetica viene presentata
come energia FPGA misurata. Le curve dense vengono prodotte sui checkpoint selezionati, non a
ogni epoca, per contenere il costo della valutazione.

## 13. Chiusura e ripresa delle augmentation

La fase si chiude al vincitore confermato, all'esaurimento del budget o all'esito negativo dei
rami ammessi. Un insuccesso viene conservato con la sua portata: non si cancella la storia per
rendere invisibili esperimenti validi ma negativi.

Il riferimento finale può essere C0, C0 con sola continuazione, una ricetta predittiva a struttura
identica, oppure un modello con routing. Si congelano insieme **struttura, inizializzazione e
procedura di training**, prima di riprendere augmentation sulla sola candidata.

Spatial erasing mantiene la propria evidenza positiva sul vecchio C0; se il modello cambia, il
suo beneficio non si assume additivo e richiede un confronto sul nuovo riferimento. Temporal
Maskout resta negativo nella configurazione già provata. Non si duplicano gli screen su B.
Un grande pretraining JEPA/EMA da zero e nuovi teacher non sono autorizzati da questa roadmap.

La base bibliografica e i limiti di confronto sono nella [review](PREDICTIVE_TEMPORAL_RESEARCH_REVIEW.md).
In particolare [F³](https://arxiv.org/abs/2509.25146) motiva l'apprendimento predittivo su eventi;
[MEOM](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f04957cc30544d62386f402e1da0b001-Abstract-Conference.html)
motiva la domanda sulla supervisione temporale. Le formule e le scelte numeriche qui fissate
sono proposte locali, non repliche né garanzie di riprodurne i guadagni.
