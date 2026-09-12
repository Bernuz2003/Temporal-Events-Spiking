# Fonti e selezione delle famiglie architetturali

**Aggiornato:** 2026-09-12

I punteggi pubblicati non sono direttamente confrontabili con la development validation locale:
molti lavori selezionano sul test ufficiale, usano crop/binning/augmentation diversi o modelli molto
più grandi. Le fonti motivano i meccanismi da testare; non forniscono una soglia da copiare.

## Evidenza che guida F e T

- [MaxFormer, NeurIPS 2025](https://arxiv.org/abs/2505.18608) e
  [codice ufficiale](https://github.com/bic-L/MaxFormer): su CIFAR10-DVS l'embedding gerarchico
  Conv-BN-MaxPool migliora lo stesso backbone con SSA. È evidenza analogica forte per ridisegnare
  l'embedding, non una garanzia su DVS-Lip. F ne usa il principio con una topologia locale più
  piccola e mantiene invariato il resto del modello.
- [Mul-free channel-wise PSN, NeurIPS 2025](https://arxiv.org/abs/2501.14490) e
  [codice ufficiale](https://github.com/Tab-ct/chwPSN): la memoria channel-wise di ordine basso è
  competitiva su DVS-Lip. Il sistema pubblicato cambia però backbone, primo layer temporale e
  readout. T isola il nucleo utile come FIR causale depthwise, senza adottare l'intero sistema.
- [Multi-Delay Mixer, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Shi_Temporal_Interaction_in_Spiking_Transformers_with_Multi-Delay_Mixer_CVPR_2026_paper.html):
  supporta interazioni temporali esplicite a ritardi multipli. L'apprendimento discreto dei ritardi
  resta fuori dal primo test; tre tap fissi nella geometria e apprendibili nei pesi sono più facili
  da attribuire e profilare.

Il FIR depthwise T da 576 coefficienti resta una candidata di compressione, ma è troppo vincolato
per rigettare da solo l'utilità di una memoria esplicita: ogni canale può soltanto filtrare la
propria storia. Il proof-of-usefulness ha quindi usato un **MIMO multi-delay causale** nei medesimi due
punti a bassa risoluzione. Per ogni ritardo `d ∈ {1,2,4}`, una matrice `C×C` proietta esclusivamente
`x[t-d]`; il percorso corrente resta identità. È una sonda locale ispirata al principio multi-delay,
non una riproduzione di MD-Mixer o chwPSN. L'inizializzazione nulla delle matrici garantisce
equivalenza iniziale con la baseline e impedisce capacità statica aggiuntiva. TCAP ha prodotto
+3,97 pp F1; T resta la compressione post-freeze e non riceve budget nella selezione prestazionale.

[Fang et al., ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Fang_Incorporating_Learnable_Membrane_Time_Constant_To_Enhance_Learning_of_Spiking_ICCV_2021_paper.html)
motiva PLIF e parametrizza `1/τ = sigmoid(w)`. La variante locale usa un `w` per feature channel
(per head nei tensori interni dell'attenzione), in tutti i LIF. Parte da `τ=2`, conserva soglia,
reset e surrogate della baseline, non aggiunge stato e rende profilabile la distribuzione di τ.

## Triage PLIF/PMSN/PSN/LMU/Mamba/GRU

| Famiglia | Evidenza utile | Rischio nel nostro protocollo | Decisione e trigger |
|---|---|---|---|
| **PSN channel-wise** | evidenza diretta DVS-Lip; memoria temporale di ordine basso | il risultato pubblicato confonde neuron model, Conv3d iniziale, backbone e readout | TCAP è positivo; T viene rivalutato soltanto come compressione post-freeze |
| **PLIF** | [Fang et al. 2021](https://arxiv.org/abs/2007.05785): costante di tempo apprendibile e minore sensibilità all'inizializzazione su benchmark neuromorfici | il run locale chiude a −0,47 pp F1 ma anticipa la performance ai prefissi | nessun nuovo training; sola diagnostica checkpoint-only B/PLIF |
| **PMSN** | confronto DVS-Lip favorevole in [Neuromorphic Sequential Arena](https://arxiv.org/abs/2505.22035) | modello pubblicato circa 9.5M, readout/dense head e protocollo diversi; dinamica parallelizzata meno naturale per streaming stateful | nessun run ora; rivalutare solo se T aiuta molto e serve una memoria temporale più lunga |
| **GRU / SpikGRU** | [SpikGRU2+](https://openaccess.thecvf.com/content/CVPR2024W/EVW/html/Dampfhoffer_Neuromorphic_Lip-Reading_With_Signed_Spiking_Gated_Recurrent_Units_CVPRW_2024_paper.html) mostra che la ricorrenza gated è forte su DVS-Lip | sistema bidirezionale da decine di milioni di parametri, 90 bin e augmentation forte | gated-v2 corretto ha fallito il gate: nessun GRU run nella discovery corrente |
| **LMU** | [LMUFormer](https://arxiv.org/abs/2402.04882) mostra memoria compatta, training parallelo e inferenza streaming su task di sequenza e speech | nessuna evidenza diretta DVS-Lip; ordine e finestra di memoria aprirebbero nuove scelte e l'integrazione richiederebbe una nuova architettura | fuori dal budget corrente; considerare soltanto se emerge una dipendenza lunga che FIR/PLIF non catturano |
| **Mamba** | modelli state-space efficaci su sequenze; [TVTA 2026](https://arxiv.org/abs/2607.08236) usa un modulo Mamba su DVS-Lip | il risultato DVS-Lip usa Mamba bidirezionale, supervisione visemica e un sistema più ampio; costo e causalità cambiano | rinviato; non è un'ablazione minima del modello corrente |

Questa graduatoria evita una matrice di neuron model: TCAP ha stabilito l'utilità del mixing
ritardato, T è rinviato e PLIF resta una diagnosi di dinamica. Gated-v2, PMSN, LMU, Mamba e GRU non
ricevono altro budget nella discovery corrente.

## Riferimenti DVS-Lip e protocollo

- [MSTP, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Tan_Multi-Grained_Spatio-Temporal_Features_Perceived_Network_for_Event-Based_Lip-Reading_CVPR_2022_paper.html):
  dataset, split officiale, Acc1/Acc2 e baseline di letteratura.
- [Codice ufficiale MSTP](https://github.com/tgc1997/event-based-lip-reading): semantica del loader;
  la funzione pubblica inverte le etichette testuali Acc1/Acc2 rispetto al paper.
- [SpikGRU-DVSLip](https://github.com/manondampfhoffer/SpikGRU-DVSLip): rappresentazione a 90 bin,
  augmentation e protocollo del modello ricorrente.
- [NeuroSeqBench](https://github.com/liyc5929/neuroseqbench): implementazioni dei neuron model e
  risultati di benchmark; non sostituisce un confronto controllato nel nostro backbone.

## Limiti delle proxy hardware

Le operazioni potenziali e gli accessi di stato sono proprietà del grafo e dell'attività osservata.
Non equivalgono a joule, latenza o area. Affermazioni sull'hardware reale richiedono una mappatura,
precisioni, gerarchia di memoria e misure su una piattaforma dichiarata.

[Horowitz, ISSCC 2014](https://doi.org/10.1109/ISSCC.2014.6757323), Fig. 1.1.9, fornisce il
riferimento aritmetico FP32 usato dalla proxy energetica v4; formule e limiti in `HARDWARE_NOTES.md`.

## Rappresentazione degli eventi: decisione del 2026-09-09

La rappresentazione non viene più trattata come un dettaglio fisso. E0 somma ON/OFF in 40 finestre
fisiche da 50 ms e perde l'ordine degli eventi dentro ogni finestra. L'aumento diretto di T non è
una soluzione economica: nel modello corrente replica quasi tutte le dinamiche LIF, le operazioni e
il traffico di stato per un numero maggiore di step.

### Fact-check TBR e Spike-TBR

Il [paper TBR originale, ICPR 2020/2021](https://fedebecat.github.io/assets/papers/innocenti2021temporal.pdf)
definisce `N` mappe binarie di occupazione: un pixel vale uno se contiene almeno un evento nel
micro-intervallo. Le `N` mappe vengono interpretate come una stringa binaria, col micro-intervallo
più recente come bit più significativo, convertite in un valore e normalizzate per `2^N−1`. La
compattazione è lossless soltanto rispetto alle mappe di occupazione alla risoluzione `Δt`; perde
polarità e molteplicità degli eventi nello stesso pixel/micro-bin.

Il [paper Spike-TBR, Pattern Recognition Letters 2025](https://flore.unifi.it/retrieve/01bfd173-dd09-4d1a-9950-bc884e6400db/2506.04817v2.pdf)
conferma per DVS-Lip:

- `N=8` in tutti gli esperimenti;
- `Δt=6,25 ms` per DVS-Lip, scelto perché lo stream è più sparso;
- quindi `ΔT=NΔt=50 ms`, coincidente esattamente con il macro-bin E0 locale;
- accuracy TBR `70,00%` e Spike-TBR-LIF `75,91%` nello stesso classificatore I3D, delta `+5,91 pp`;
- `β=0,9` selezionato su validation per DVS-Lip e soglia LIF `1,1`;
- LIF `75,91%`, RecLIF `74,45%`, LRLIF `71,53%`, PLIF `74,45%` sul dato pulito.

Il delta è accuracy del protocollo del paper, non Macro-F1 locale, e non è confrontabile coi nostri
punteggi development. Il paper non fornisce un repository ufficiale e l'algoritmo lascia non
completamente specificati `w(p)` e la continuità della membrana tra finestre `ΔT`. La variante
locale Spike-TBR è pertanto registrata come **paper-aligned reconstruction**: ignora la polarità,
usa i count per micro-bin come input, reset hard e reinizializza la membrana a ogni macro-finestra,
coerentemente con l'inizializzazione per `ΔT` dell'algoritmo. Queste scelte sono salvate nella config
e nei metadata e non saranno oggetto di sweep.

La verifica empirica locale rende questa ambiguità materiale: sui primi 64 sample validation la
ricostruzione emette in media 808,5 voxel macro non nulli, contro 7.594,6 del TBR canonico. Una
simulazione checkpoint-free che mantiene la membrana fra macro-finestre sale a 2.123,7 voxel, ma
resta molto più sparsa e non è autorizzata come nuovo candidato: scegliere la policy dopo aver visto
il fallimento sarebbe tuning della ricostruzione, non replica del risultato pubblicato.

I due run usano F come substrato. Entrambi producono `[40,1,H,W]`; il Transformer continua a
elaborare 40 step. F+TBR misura il valore del timing intra-bin compresso; F+Spike-TBR misura il
valore aggiunto dal filtro LIF a parità di shape e backbone. Non si usa una variante ON/OFF locale,
perché non sarebbe la rappresentazione alla quale si riferisce l'evidenza pubblicata.

### Multi-granularity

[MSTP, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Tan_Multi-Grained_Spatio-Temporal_Features_Perceived_Network_for_Event-Based_Lip-Reading_CVPR_2022_paper.pdf)
fornisce evidenza diretta che DVS-Lip beneficia di rami a granularità diversa: il ramo a basso frame
rate conserva struttura spaziale completa e quello ad alto frame rate privilegia dettaglio
temporale con rappresentazione spaziale più economica, poi un message-flow integra le feature.
Questo è diverso da TBR: TBR comprime deterministicamente otto occupazioni in un valore prima del
modello; un ramo multi-granular elabora esplicitamente i micro-step e apprende la compressione.

L'ablazione pubblicata è particolarmente utile: low-rate 30 bin raggiunge 69,57% accuracy,
high-rate 210 bin 69,49%, la combinazione senza Multi-Scale Feature Relation Module 71,11% e MSTP
completo 72,10%. I due rami hanno forza individuale simile ma feature complementari; il guadagno
non deriva dal semplice aumento uniforme del numero di frame. Il
[codice ufficiale](https://github.com/tgc1997/event-based-lip-reading) conferma due tensori distinti,
un ramo high-rate assottigliato nei canali e convoluzioni temporali strided che riallineano le
feature al clock low-rate prima della fusione. Quel sistema usa però voxel endpoint-normalizzati,
ResNet-18 e GRU bidirezionale; non viene copiato integralmente.

L'implementazione locale usa un solo encoder `multigranular_count_frame` e un solo ramo
parametrico. La prova di capacità conserva E0 e aggiunge count ON/OFF causali
`[320,2,32,32]`, poi applica downsampling spaziale appreso, riduzione temporale MIMO 8:1 e fusione
residua appresa; ha 480.036 parametri e resta sotto B. Il controllo Lite configura lo stesso codice
con `[320,2,16,16]`, riduzione depthwise e somma diretta, per 433.188 parametri. Questa coppia
separa la verifica del principio dal suo limite di compressione senza introdurre due moduli o due
rappresentazioni differenti.

### Alternative rinviate

[Gehrig et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Gehrig_End-to-End_Learning_of_Representations_for_Asynchronous_Event-Based_Data_ICCV_2019_paper.html)
formalizzano Event Spike Tensor/voxel grid come misura e kernel temporale differenziabili e mostrano
che la rappresentazione appresa può migliorare recognition e optical flow. Questo sostiene l'uso di
basi temporali più informative del conteggio, ma un MLP per evento e vari kernel aggiungerebbero
una nuova famiglia da ottimizzare.

[HATS, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sironi_HATS_Histograms_of_CVPR_2018_paper.pdf)
media time surfaces locali per ottenere una rappresentazione compatta e robusta con memoria.
[TORE, TPAMI 2023](https://doi.org/10.1109/TPAMI.2022.3172212) conserva in FIFO i K timestamp più
recenti per pixel/polarità, mentre
[TAF](https://arxiv.org/abs/2208.11602) campiona le ultime K posizioni temporali non nulle e le fonde
nei canali. Sono soluzioni efficienti quando la frequenza locale degli eventi varia, ma introducono
K, decadimenti/log-clipping, stato per pixel e una semantica streaming nuova. Restano seconde
candidate se un probe minimale dimostra che E0 comprime troppo.

[Matrix-LSTM, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650137.pdf)
apprende una superficie ricorrente per pixel e ha migliorato classification e optical flow rispetto
a rappresentazioni manuali. È più espressivo, ma richiede stato LSTM spaziale, kernel specializzati
e un secondo sistema ricorrente davanti alla SNN; il rapporto informazione/run e il costo hardware
sono peggiori per la fase corrente.

E1 phase-count e duration-normalized restano implementazione/probe sospesi: il primo è una
statistica locale non supportata direttamente su DVS-Lip, il secondo usa endpoint oracle. Nessuno
dei due riceve un full mentre sono disponibili TBR e Spike-TBR.

## Triage finale prima del freeze

| Famiglia | Evidenza | Decisione corrente |
|---|---|---|
| High-frequency local mixing | [MaxFormer](https://papers.neurips.cc/paper_files/paper/2025/hash/956834836f36dd07df7064ff42ca69f2-Abstract-Conference.html) mostra che MaxPool e depthwise convolution contrastano il bias low-pass delle SNN; [HFR-Lip](https://doi.org/10.1016/j.ins.2025.123026) individua direttamente su DVS-Lip la perdita di bordi e micro-deformazioni | **unico probe condizionale ammesso**, nel primo stage di F+TCAP, se MG+TCAP non raggiunge la soglia |
| PMSN | [NSA](https://www.ijcai.org/proceedings/2025/0544.pdf) riporta su DVS-Lip 57,43% contro 17,83% del LIF nello stesso MLP | non trasferibile come singola ablation: usa 200 bin, crop 88×88, last-step readout e sostituisce la dinamica neuronale dell'intera rete; inoltre la proxy pubblicata costa 6,8× il LIF |
| chwPSN / MD-Mixer | evidenza diretta o forte sul mixing temporale esplicito | TCAP ha già validato il principio; la versione depthwise appartiene alla compressione, mentre ritardi discreti soft-to-hard aprirebbero un nuovo tuning |
| SpikGRU2+ | forte risultato DVS-Lip | backbone, 90 bin, readout bidirezionale e augmentation cambiano insieme; gated-v2 locale è negativo |
| MTGA, graph, TORE/TAF, EST, Matrix-LSTM | preservano timing fine | richiedono una nuova rappresentazione e nuovi iperparametri; MG ha già fornito il probe positivo a minor costo di sviluppo |
| LMU, Mamba, S4D/GSU | forti su sequenze lunghe | nessun confronto controllato nello stesso backbone DVS-Lip e integrazione non minimale; rinviati |

Il collo residuo più concreto è spaziale: F+TCAP ottiene 44,42% su Acc1 e 61,15% su Acc2; MG
migliora F di 3,54 pp su Acc1 ma solo 0,60 pp su Acc2. Questo rende il probe high-frequency più
mirato di un altro temporal core. Non viene implementato finché MG+TCAP non determina quale
substrato architetturale debba essere congelato.
