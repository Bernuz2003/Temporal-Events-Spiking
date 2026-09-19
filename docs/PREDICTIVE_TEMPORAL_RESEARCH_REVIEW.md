# Valutazione scientifica: apprendimento predittivo e memoria temporale condizionale

**Data:** 2026-09-19. **Stato:** analisi per discussione; nessuna nuova architettura, ricetta o
campagna di training approvata da questo documento.

**Seguito operativo:** le considerazioni sono ora tradotte nella
[`PREDICTIVE_TEMPORAL_ROADMAP.md`](PREDICTIVE_TEMPORAL_ROADMAP.md). Questa review conserva il
ruolo di valutazione critica; ordine, soglie e budget attivi sono nel documento operativo.

Il report valuta la conversazione allegata su Cross-Resolution Predictive Learning (CRP),
Surprise-TCAP e TCAP dinamico, confrontandola con gli artifact, il codice corrente e fonti
primarie. Le proposte restano ipotesi. L'architettura congelata di riferimento rimane
F+DWC-3+TCAP-d8. Non vengono introdotte implementazioni o istruzioni operative.

## 1. Giudizio complessivo e valore del lavoro già svolto

La direzione è scientificamente plausibile: sfruttare meglio la struttura temporale è coerente
con il problema e con i risultati. La conversazione, però, sovrastima sia la forza delle
premesse empiriche sia la probabilità di successo e l'originalità delle soluzioni proposte.
Una narrazione coerente non sostituisce l'identificazione del meccanismo che produce il beneficio.

Non condivido la descrizione dei miglioramenti ottenuti come marginali. Dai `summary.json`:

| DVS-Lip, development validation | Seed | Macro-F1 % | Accuracy % | Parametri |
|---|---:|---:|---:|---:|
| B | 42 | 44,15 | 44,81 | 500.708 |
| F+TCAP, riferimento precedente | 42 | 52,36 | 52,79 | 492.516 |
| F+MG-Cap+TCAP | 42 | 53,15 | 53,52 | 541.476 |
| F+DWC-3+TCAP-d8 | 42 | 55,18 | 55,53 | 501.028 |
| F+DWC-3+TCAP-d8 | 43 | 54,88 | 55,09 | 501.028 |
| F+DWC-3+TCAP-d8 | 44 | 56,56 | 56,73 | 501.028 |

Il confronto appaiato seed 42 con B vale **+11,02 pp F1**, con appena 320 parametri in più.
La terna della candidata raggiunge **55,54 ± 0,90% F1** (deviazione standard campionaria).
Il trasferimento DVS-Gesture seed 42 raggiunge **88,81% F1 e 89,39% accuracy**; non costituisce
ancora una conferma multi-seed su quel dataset. Questi sono risultati di validation, non
risultati sull'official test.

Occorre separare tre contributi: una formula nuova; una combinazione motivata e verificata
di metodi esistenti; una caratterizzazione sperimentale utile di accuratezza, latenza e costo.
Il secondo e il terzo hanno valore scientifico, particolarmente in una tesi magistrale.
Un modulo aggiuntivo non rende automaticamente il lavoro più originale o più solido.

Provenienza: [B](../artifacts/dvslip_e0__20260825_211710__seed42/summary.json),
[F+TCAP](../artifacts/dvslip_f_temporal_capacity__20260909_124154_088394__seed42/summary.json),
[F+MG+TCAP](../artifacts/dvslip_f_multigranular_temporal_capacity__20260912_135802_779469__seed42/summary.json),
[congelata 42](../artifacts/dvslip_f_tcap_stage1_dwc3_d8__20260914_093427_434930__seed42/summary.json),
[43](../artifacts/dvslip_f_tcap_stage1_dwc3_d8__20260914_225132_054396__seed43/summary.json),
[44](../artifacts/dvslip_f_tcap_stage1_dwc3_d8__20260914_225137_788814__seed44/summary.json),
[DVS-Gesture](../artifacts/dvsgesture_f_tcap_stage1_dwc3_d8__20260915_094224_852880__seed42/summary.json).
I profili storici e i relativi limiti restano nel [ledger](EXPERIMENT_LEDGER.md); nessuna nuova
misura hardware è stata prodotta per questo report.

## 2. Cosa dimostrano davvero TCAP, MG e PLIF

| Osservazione | Conclusione sostenuta | Conclusione non ancora sostenuta |
|---|---|---|
| TCAP migliora la classificazione | L'integrazione temporale esplicita è utile nel nostro setup | Predire feature future migliorerà necessariamente la classificazione |
| Il modello dipende dai tap lunghi | La soluzione appresa usa quel contesto temporale | Quei tap rappresentano un predittore ottimale o richiedono un router dinamico |
| MG migliora alcuni risultati e prefissi | Il ramo aggiunto offre un segnale promettente | Il vantaggio deriva unicamente dal timing fine; il ramo isolato è un buon teacher |
| PLIF migliora decisioni precoci ma non il risultato finale | Accuratezza finale e qualità dei prefissi sono obiettivi distinti | Una predictive loss recupererà automaticamente il vantaggio PLIF a fine finestra |
| I ritardi apprendibili non migliorano significativamente d8 | Quella formulazione non ha mostrato un vantaggio convincente | Non esistono ritardi statici adeguati e serve una selezione dipendente dal contenuto |

Per MG, il confronto con F+TCAP vale soltanto **+0,795 pp F1**. L'analisi appaiata già registrata
riporta IC95% `[-1,16; +2,71]` pp e un costo di training circa 2,9 volte maggiore. Questo non
prova l'assenza di utilità; rende però troppo forte la frase «sappiamo che il teacher fine
contiene l'informazione discriminativa che manca allo student». Cambiano anche capacità,
elaborazione e fusione, non soltanto la risoluzione temporale. Il riferimento congelato attuale
è inoltre più forte del vecchio F+TCAP usato per quel confronto.

L'azzeramento di un tap in una rete già addestrata misura dipendenza e co-adattamento. Non equivale
a confrontare due modelli riaddestrati e non identifica da solo una carenza architetturale.

## 3. Verifica della letteratura e limiti delle analogie

| Fonte primaria | Risultato pertinente | Limite dell'uso nella conversazione |
|---|---|---|
| [F³](https://arxiv.org/abs/2509.25146) | Apprende rappresentazioni predicendo eventi futuri; risultati su flow, segmentazione e profondità | Sostiene la motivazione predittiva, non un guadagno dimostrato della nostra loss cross-resolution nella classificazione DVS-Lip |
| [TESPEC, ICCV 2025](https://arxiv.org/abs/2508.00913) | Pretraining ricorrente con storia lunga e ricostruzione di video pseudo-grayscale | Non è una dimostrazione di previsione di feature MG; task e target sono diversi |
| [SPICE, TU Delft](https://repository.tudelft.nl/record/uuid:788c2e55-601b-45ae-aa53-8ea45001a3c8) | Predizione di latenti futuri per eventi e obiettivo contrastivo su regioni attive | È una tesi magistrale del 2025, non un articolo di conferenza; indica anche limiti di stabilità temporale e organizzazione semantica |
| [I-JEPA, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html) | Predizione di rappresentazioni di blocchi di immagine da un contesto; scelta di contesto e target decisiva | Una loss fra feature con stop-gradient non basta a riprodurre il metodo o le sue proprietà |
| [Predictive Coding Light](https://pmc.ncbi.nlm.nih.gov/articles/PMC12500994/) | Inibizione appresa che sopprime spike prevedibili; compromesso informazione/attività | Non dimostra che attenuare la memoria quando l'errore è alto aumenti l'accuracy; riporta risparmi di spike con perdita piccola/moderata di prestazione |
| [MD-Mixer, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Shi_Temporal_Interaction_in_Spiking_Transformers_with_Multi-Delay_Mixer_CVPR_2026_paper.html) | Mixing multi-delay per interazioni temporali nei transformer spiking | Conferma che il territorio è già esplorato; il grado di novità di un router richiede confronto puntuale delle equazioni |
| [MEOM, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f04957cc30544d62386f402e1da0b001-Abstract-Conference.html) | Distillazione con prospettive temporali e allineamento progressivo delle predizioni troncate | È prior art vicino alla supervisione temporale e dei prefissi, non evidenza specifica di CRP |
| [D3D, WACV 2020](https://openaccess.thecvf.com/content_WACV_2020/papers/Stroud_D3D_Distilled_3D_Networks_for_Video_Action_Recognition_WACV_2020_paper.pdf) | Distillazione di informazione di movimento da un ramo usato soltanto nel training | Usare informazione privilegiata durante il training senza quel ramo a inference è un principio già noto |
| [Selective Kernel Networks, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Selective_Kernel_Networks_CVPR_2019_paper.html) | Selezione dipendente dall'input fra scale del campo recettivo | Il principio generale di pesare dinamicamente scale diverse non è nuovo |

Le fonti sostengono la plausibilità delle famiglie, non una previsione affidabile di +N punti.
Questa verifica non è una ricerca esaustiva di anteriorità: non autorizza affermazioni «primo
metodo» o «intersezione mai esplorata». In particolare non assumiamo che ogni weighting descritto
come adaptive in letteratura sia dipendente dall'input: parametri appresi ma fissi a inference e
pesi ricalcolati sul campione sono meccanismi differenti.

## 4. Cross-resolution predictive learning: promettente, ma va definito meglio

### 4.1 Ipotesi difendibile

La domanda interessante è: **una supervisione temporale più ricca, disponibile solo durante
il training, può migliorare ciò che il modello coarse conserva per classificare?**

La forma proposta è plausibile:

\[
\mathcal L=\mathcal L_{CE}+\lambda D(P(c_{\le t}),\operatorname{sg}(z^f_{t+h})).
\]

Rimuovendo teacher e predictor, la topologia e il costo nominale dense a inference restano quelli
del modello congelato. È un vantaggio reale. Il costo di training, però, cresce e l'energia
dipendente dall'attività va rimisurata: pesi diversi possono cambiare firing rate e SOP.

### 4.2 Non recupera informazione irrecuperabile

Se due stream producono esattamente lo stesso input coarse fino a t, il modello causale non
può sapere quale dei due micro-pattern sia avvenuto, né quale futuro non deterministico seguirà.
La supervisione fine può insegnare regolarità statistiche e preservare indizi utili già presenti;
non annulla la perdita informativa dell'encoding.

Con una loss quadratica, il predittore ottimale stima una media condizionata del target.
Futuri multimodali possono quindi essere mediati: proprio le differenze fini utili a distinguere
due parole possono sparire. Rendere un target facile da predire non equivale a renderlo utile
alla classificazione. Predittività, rilevanza semantica e costo sono tre criteri distinti.

### 4.3 Il teacher è il punto più fragile

Il ramo fine MG è stato addestrato con il ramo coarse e la fusione: potrebbe rappresentare
un residuo complementare, non una rappresentazione semanticamente completa e trasferibile.
Una feature pre-LIF è un target continuo ragionevole, ma «continuo e signed» non implica
informativo: può essere dominato da offset BN, ampiezza, sfondo e componenti poco discriminative.
La cosine loss non elimina questi problemi e ignora parte dell'informazione di ampiezza.

Un teacher complessivamente meno accurato può comunque insegnare competenze complementari.
Non è quindi necessario che MG superi il finalista, ma è necessario evitare di dedurre la sua
qualità come teacher dal solo risultato del modello fuso. Anche la medesima shape delle feature
non garantisce allineamento semantico, spaziale o delle scale temporali.

Poiché il checkpoint MG è supervisionato, il metodo sarebbe **distillazione predittiva con
informazione privilegiata**, non apprendimento completamente self-supervised. L'auxiliary loss
può non usare esplicitamente etichette, ma il teacher ne incorpora l'informazione. Un teacher
congelato offre target stabili; uno jointly trained con stop-gradient non è automaticamente
protetto dal collasso. Un EMA fra architetture coarse e fine differenti non è una sostituzione
immediata del teacher.

### 4.4 Quale contributo si potrebbe rivendicare

Il confronto futuro-fine contro presente-fine distingue anticipazione e distillazione simultanea.
Per sostenere anche che sia essenziale la risoluzione fine, serve distinguere quel beneficio
da una predizione futura alla stessa risoluzione coarse. Non sono tutti training da avviare ora:
sono i controlli richiesti dalle eventuali affermazioni finali.

Se funziona soltanto il target presente, avremo un risultato di distillazione valido; non una
conferma dell'anticipazione. Se funziona ugualmente il futuro coarse, potremo conservare la
soluzione meno costosa e restringere correttamente la rivendicazione scientifica.

## 5. Surprise-TCAP: il problema centrale è il significato dell'errore

È corretto separare il predittore dalle matrici discriminative di TCAP: non c'è ragione per cui
\(\sum_dW_dx_{t-d}\) debba ricostruire \(x_t\). Tuttavia il nuovo predittore deve avere un obiettivo
che renda interpretabile il residuo. Se ottimizzato soltanto attraverso la classificazione,
può diventare un altro insieme di feature per il gate; chiamarlo prediction error non basta
a dimostrare che predice.

Il predittore depthwise proposto è economico ma limitato: vede lo stesso canale e la stessa
posizione nei ritardi precedenti. Un bordo che si sposta può apparire imprevedibile per quel
predittore anche quando il movimento è regolare. Se i coefficienti sono non negativi e sommano
a uno, la previsione è una combinazione convessa del passato, incapace di estrapolare oltre
il suo intervallo. Il solo vincolo di somma uno, senza non negatività, non garantisce tale
interpretazione né stabilità.

### 5.1 «Alta surprise → meno memoria» non è una regola generale

| Origine di un errore alto | Possibile ruolo della memoria |
|---|---|
| Cambio reale di dinamica | Alcune parti della storia possono diventare meno pertinenti |
| Rumore o artefatto nel presente | La storia può proteggere la decisione |
| Movimento spaziale non rappresentabile dal predittore | Il passato può restare utile; è il predittore a essere inadeguato |
| Transizione discriminativa tra gesti/fonemi | Il confronto con ciò che precede può essere essenziale |
| Variazione di scala delle feature | Non implica alcun cambiamento della memoria utile |

Il residuo assoluto è una misura di discrepanza, non automaticamente sorpresa probabilistica
o incertezza calibrata. La media su canali e spazio può inoltre diluire micro-movimenti locali.
Il gate monotono decrescente impone dunque una scelta forte proprio dove l'evidenza manca.

La domanda da affinare è: **l'errore predittivo contiene informazione sull'utilità del contesto
temporale, oltre a quella già contenuta nell'attività corrente?** Soltanto una risposta positiva
giustificherebbe un controllo della memoria basato su quell'errore. La relazione dovrebbe
distinguere affidabilità della previsione, affidabilità dell'osservazione e rilevanza dei ritardi;
queste tre quantità non coincidono.

### 5.2 Non è ancora dimostrato un risparmio computazionale

Un gate soft moltiplicato per un contributo già calcolato non evita quel calcolo. Servono comunque
il presente, la previsione e il confronto; non si risparmia il front-end che ha prodotto x.
La versione proposta mantiene inoltre il contributo della memoria vicino a uno quando il segnale
è prevedibile: non realizza la precedente promessa di saltare computazione ridondante proprio
nei periodi prevedibili.

I 768 coefficienti del predittore sono pochi parametri, ma vengono applicati su spazio e tempo.
Vanno contati MAC, riduzioni, moltiplicazioni del gate, traffico e buffer. Riduzione degli spike,
riduzione delle operazioni eseguite e riduzione della latenza GPU sono risultati distinti.
La proxy Horowitz deve esplicitare quali operazioni condizionali l'hardware potrebbe evitare.

Usare un residuo per modulare una rete è compatibile con un'ispirazione predittiva, ma non basta
a dimostrare equivalenza a un modello normativo di predictive coding. Anche la membrana LIF
non diventa formalmente un errore di predizione senza definire stato predetto, decoder e dinamica.

## 6. TCAP dinamico: plausibile, con originalità più circoscritta

La forma \(y_t=x_t+\sum_dg_d(x_t)W_dx_{t-d}\) pone una domanda chiara: l'utilità relativa delle
scale temporali varia con il contenuto? È un'estensione ragionevole, ma appartiene alla famiglia
già ampia delle convoluzioni condizionali e della selezione adattiva delle scale.

L'inizializzazione \(g_d=1\) può preservare la funzione di un checkpoint TCAP e ridurre la
perturbazione iniziale. Non garantisce però un'ottimizzazione utile. Se le matrici temporali
sono inizializzate a zero, inizialmente anche il gradiente dei router attraverso quel contributo
è nullo; ciò non implica un blocco permanente, ma va distinto dal caso di un checkpoint allenato.
Analogamente un gate sorpresa con ampiezza zero ritarda l'apprendimento di alcuni suoi parametri.

Un miglioramento potrebbe derivare da ricalibrazione quasi costante o capacità aggiuntiva.
Per sostenere l'adattività serve evidenza che la dipendenza dal campione/tempo conti davvero;
visualizzare gate diversi da uno non basta. Le ampiezze dei gate e delle matrici possono
compensarsi, quindi non sono direttamente misure identificabili di importanza dei ritardi.

Il pooling globale è un possibile limite per movimenti localizzati. Tuttavia introdurre subito
router per pixel, canale e tap creerebbe una nuova ricerca architetturale: la complessità va
giustificata dall'ipotesi, non dalla disponibilità di parametri.

## 7. Un rischio concreto nel codice: causalità delle statistiche di normalizzazione

In [`layers.py`](../src/etsr/models/layers.py), `_time_distributed` unisce tempo e batch prima
della BatchNorm. In modalità training, le statistiche di `BatchNorm2d` dipendono quindi anche
da timestep successivi. `FineTemporalBranch.temporal_bn` aggrega a sua volta lungo l'asse
temporale della sequenza ridotta.

Per la classificazione offline di sequenze complete questo comportamento non invalida di per sé
i risultati correnti. A inference, con statistiche running fissate, non implica accesso al futuro.
Ma per affermare «il contesto fino a t predice il futuro» durante il training è un canale
indiretto di informazione futura da considerare esplicitamente. Una convoluzione causale e uno
stop-gradient non lo eliminano. Un teacher con pesi congelati ma rimasto in modalità training
può inoltre aggiornare le statistiche e dipendere dagli altri timestep.

È una condizione metodologica da risolvere prima di una futura implementazione, non un motivo
per riscrivere ora il backbone. Anche target futuri ottenuti da feature ricorrenti possono
condividere molta storia con il contesto: predirli bene può riflettere persistenza anziché
anticipazione di informazione nuova.

## 8. Probe e coda temporale: utili, senza attribuire loro potere eccessivo

I probe congelati proposti sono ragionevoli per controllare le premesse, ma:

- Un probe lineare fallito limita quella rappresentazione e quel readout; non prova che una
  rappresentazione riappresa non possa trarre beneficio dalla loss.
- Una cosine similarity elevata può derivare da offset condivisi, persistenza e coda quasi
  deterministica. Va confrontata con predizioni banali e con la variabilità effettiva dei target.
- Un predittore che batte la persistenza mostra una relazione statistica utile, non ancora
  un beneficio discriminativo o un risparmio energetico.
- Scegliere l'orizzonte sulla validation è comunque selezione sperimentale, anche senza usare
  le etichette di classificazione. Il test resta escluso; finestre sovrapposte della stessa
  sequenza non devono essere distribuite tra fit e valutazione del probe.
- Un riferimento che accede alla feature fine precedente non è disponibile allo student coarse
  a inference: può essere un confronto diagnostico privilegiato, non un baseline deployable.

La coda post-evento merita attenzione, ma non è semplicemente «predici zero»: gli stati e le
feature interne possono continuare a evolvere, come mostrato dalla diagnostica B/PLIF.
Limitare la loss ai target precedenti all'ultimo evento è una scelta possibile, non neutrale;
può escludere transizioni e memoria utili. Usare `last_event+8` perché otto è il tap massimo
non dimostra che 400 ms sia il tempo di assestamento corretto.

La valutazione dovrebbe distinguere attività, transizioni e coda, senza ottimizzare a posteriori
una soglia sulla durata tipica DVS-Lip. Usare l'endpoint soltanto per costruire una loss offline
non rende automaticamente non causale il modello deployato; introduce però una scelta di
supervisione e una pesatura temporale da dichiarare, con trasferibilità da verificare.

## 9. Come affinare la direzione scientifica

La formulazione più difendibile è studiare **quale informazione temporale valga la pena conservare
e utilizzare a un costo di inference fissato**. In questo quadro esistono due domande autonome:

1. Supervisione: target temporali più informativi migliorano la rappresentazione coarse?
2. Inference: la rilevanza della storia cambia in modo sfruttabile con il contenuto e la sua
   affidabilità?

CRP e routing possono combinarsi, ma non sono ortogonali in senso sperimentale: la loss cambia
le feature su cui operano predittore e gate. La moltiplicazione di un gate globale sorpresa e
di gate per tap introduce anche ambiguità di scala. Non assumiamo complementarità prima dei dati.

Una direzione alternativa coerente con PLIF è **rendere prima disponibile una decisione utile**
attraverso supervisione temporale o distillazione dei prefissi. MEOM è un antecedente pertinente.
Qui il beneficio atteso riguarda anzitutto curva accuratezza-latenza e dipendenza dalla coda,
non necessariamente il massimo score finale. Prefissi realmente ambigui non possono essere
forzati a identificare con certezza la parola completa: il target futuro è informazione
privilegiata, non evidenza già osservabile.

Per Surprise-TCAP, l'affinamento prioritario è concettuale: collegare la modulazione al **valore
discriminativo del contesto**, anziché postulare che più errore significhi meno memoria.
Non occorre ora aggiungere un nuovo modulo per ogni possibile nozione di incertezza.

## 10. Priorità raccomandate e criteri di interpretazione

| Direzione | Potenziale | Rischio principale | Giudizio attuale |
|---|---|---|---|
| Distillazione/predizione cross-resolution | Buona domanda di ricerca, struttura deployata invariata | Teacher e target non necessariamente utili; possibile leakage temporale in training | Prima priorità di approfondimento scientifico, senza promessa di guadagno |
| TCAP dipendente dal contenuto | Estensione diretta del meccanismo più efficace | Beneficio piccolo o dovuto a ricalibrazione; novità incrementale | Candidato ragionevole se si decide di riaprire limitatamente la struttura |
| Surprise-TCAP | Potenziale contributo meccanicistico | Relazione errore→memoria attualmente non giustificata | Da riformulare prima di considerarlo pronto per training |
| Supervisione dei prefissi | Coerente con diagnostica PLIF e costo della coda | Ambiguità dei prefissi e obiettivo diverso dallo score finale | Alternativa pertinente se latenza/PrefixAUC diventano obiettivi espliciti |
| Combinazione delle tre idee | Possibili sinergie | Interazioni, attribuzione debole, moltiplicazione del budget | Prematura |
| Error-LIF, STDP, nuovi SSM/LMU | Possibili altri programmi di ricerca | Ampio cambio di paradigma e controlli aggiuntivi | Nessuna evidenza locale sufficiente a riaprirli ora |

La lista allegata di 6–8 full training, oltre ai probe, può diventare molto più ampia includendo
controlli e repliche. Non la adottiamo come pacchetto. La disponibilità di calcolo non sostituisce
un ordine di priorità, e l'assenza di una novità algoritmica radicale non obbliga a espandere il
progetto.

Un'eventuale nuova campagna dovrà attribuire il risultato al meccanismo dichiarato: teacher
versus predizione; capacità versus adattività; loss ausiliaria versus routing. Un warm-start
richiede un riferimento che riceva anch'esso l'ulteriore training, altrimenti tempo e metodo
si confondono. I candidati promettenti richiedono conferma fra seed, una ricetta di confronto
fissa e profiling del proprio checkpoint. Il costo del teacher conta nel budget di ricerca;
MAC, SOP, firing rate, stato e proxy Horowitz contano nel budget di deployment.

Il contributo eventualmente rivendicabile sarà quello dimostrato. CRP migliore della distillazione
simultanea sostiene il valore dell'anticipazione; un router migliore dei pesi costanti sostiene
il valore della condizionalità; errori predittivi associati a scelte temporali utili sostengono
un meccanismo di controllo della memoria. Un aumento dello score, da solo, non dimostra tutti
questi passaggi.

**Raccomandazione:** preservare il riferimento congelato e sviluppare prima una domanda precisa
sulla supervisione temporale; considerare TCAP dinamico come seconda ipotesi; non adottare
ancora la regola «alta sorpresa → soppressione del passato». Il risultato scientifico più forte
sarebbe identificare quando informazione temporale aggiuntiva aiuta, quando è ridondante e
quanto costa sfruttarla, anche se la soluzione finale risultasse più semplice della narrativa
Predict–Compare–Route.
