# Roadmap di validazione e raffinamento

**Definita:** 2026-09-13

Questa fase inizia dopo la chiusura della discovery architetturale e mantiene separati tre quesiti:

1. la superiorità strutturale è riproducibile fra seed e dataset?
2. quale ricetta supervisionata sfrutta meglio la struttura congelata?
3. il pretraining auto-supervisionato aggiunge valore oltre il miglior riferimento supervisionato?

L'official test DVS-Lip resta escluso fino alla scelta definitiva di struttura e ricetta.

## Punto di partenza

Il record development a seed 42 è **F+MG-Cap+TCAP**, con 53,52% accuracy e 53,15% Macro-F1.
Il finalista strutturale corrente è però **F+TCAP**, con 52,79/52,36% e un profilo nettamente più
economico. Il guadagno marginale di MG sul finalista è 0,73 pp accuracy e 0,80 pp F1; il bootstrap
appaiato stratificato per classe dà un IC95% del delta F1 di `[-1,16; +2,71]` pp e McNemar esatto
sull'accuracy dà `p=0,479`. Il run non raggiunge quindi la soglia preregistrata di +2 pp.

Rispetto a F+TCAP, il combinato richiede +9,94% parametri, +34,16% stato persistente, +23,08% MAC
multivalore, +25,89% SOP potenziali, +22,45% nella proxy Horowitz ad attività e circa 2,9 volte il
tempo di training. MG non viene portato nella campagna multi-seed principale. Resta un risultato
positivo di rappresentazione e latenza: aumenta il F1-PrefixAUC di 2,34 pp e il F1 a 1 s di 5,05
pp rispetto a F+TCAP.

## Chiusura controllata della discovery

Restano al massimo due verifiche: il probe high-frequency è autorizzato dall'esito MG+TCAP; il run
con un ritardo aggiuntivo dipende invece dalla diagnostica checkpoint-only.

### 1. Probe high-frequency nel primo stage

F usa già il max-pooling nel patch embedding, cioè una delle due correzioni proposte da
[MaxFormer](https://papers.neurips.cc/paper_files/paper/2025/hash/956834836f36dd07df7064ff42ca69f2-Abstract-Conference.html).
Il lavoro sostituisce inoltre la self-attention dei primi stage con una depthwise convolution 3×3;
su CIFAR10-DVS la sua ablation favorisce DWC-3 rispetto a DWC-1 e DWC-5. La motivazione è
coerente con [HFR-Lip](https://doi.org/10.1016/j.ins.2025.123026), che individua direttamente su
DVS-Lip bordi e micro-deformazioni labiali come informazione discriminativa persa.

Si ammette un solo candidato: F+TCAP, E0 invariata, `stage1` locale DWC-3 al posto di Token-QK,
stesso residual block, stesso stage 2 e stessa recipe. Non si provano kernel, posizioni o fusioni
alternative. Si promuove se guadagna almeno 2 pp F1 oppure, come candidato Pareto, se resta entro
0,5 pp F1 riducendo in modo misurato SOP ed energia. In caso contrario si congela F+TCAP.

### 2. Diagnostica dei ritardi TCAP

Prima di aggiungere tap si usa il checkpoint F+TCAP esistente e si azzera, a turno, il contributo
dei ritardi 1, 2 e 4 in tutti i mixer e poi l'intera storia. Si misurano F1, accuracy, Acc1/Acc2 e
curve a prefisso.
È una diagnostica senza training e mostra se il tap più lungo, 200 ms, è ancora al bordo della
memoria utile.

Un solo run `[1,2,4,8]` è autorizzato se rimuovere `d=4` costa almeno 1 pp F1 e il suo contributo
non è inferiore a quello di `d=2`. Il nuovo tap corrisponde a 400 ms, intervallo coerente con la
persistenza osservata nella diagnostica PLIF. Si applica la stessa soglia di promozione di +2 pp.
L'apprendimento continuo/discreto dei ritardi, pur supportato su task speech da
[DCLS](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4df1cc5a7528b7197ad8ae76ff30107a-Abstract-Conference.html),
non faceva parte dei due probe iniziali. Si ammette ora un solo confronto aggiuntivo,
**DWC-3 + TCAP con quattro ritardi apprendibili per canale**, inizializzati a `[1,2,4,8]` e
vincolati a `1..8` bin. Le matrici MIMO, la rappresentazione e la recipe restano quelle di
DWC-3+d8; il confronto diretto è con quel run. Il training usa una distribuzione triangolare con
temperatura decrescente `τ(p)=0,501+(4−0,501)[(1+cos(πp))/2]²`; selezione del best, validation e
profiling usano sempre i ritardi interi. Il [supplementary MD-Mixer](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Shi_Temporal_Interaction_in_CVPR_2026_supplemental.pdf)
scrive un fattore `(1+cos)²/2`, incompatibile a `p=0` con il proprio `τmax`: qui si usa `/4` per
conservare la forma quadratica e rispettare gli estremi dichiarati. La traiettoria dei ritardi è
registrata a ogni epoca in `learned_delay_trajectory.csv`.
La [ablazione MD-Mixer](https://openaccess.thecvf.com/content/CVPR2026/papers/Shi_Temporal_Interaction_in_Spiking_Transformers_with_Multi-Delay_Mixer_CVPR_2026_paper.pdf)
motiva la prova, ma confronta i ritardi appresi con ritardi casuali, non con i nostri tap geometrici.
Un solo risultato seed 42 resta esplorativo: non congela l'architettura senza conferma multi-seed.

**Esito diagnostico.** Sul checkpoint F+TCAP seed 42, l'azzeramento di `d=1/2/4` riduce il
Macro-F1 rispettivamente di 40,17/49,79/51,60 pp; senza tutta la storia il calo è 52,16 pp. Il tap
4 ha inoltre la norma maggiore in entrambi i mixer. Il criterio è soddisfatto e autorizza il solo
run d8; l'ablation mostra co-adattamento e dipendenza, non garantisce un guadagno marginale.

Se entrambi i probe superano la soglia, si consente una sola combinazione. Se nessuno la supera,
la discovery termina senza altre varianti di MG, PLIF, neuron model, rappresentazione o readout.

## Fase A — conferma multi-seed su DVS-Lip

Si fissano prima dei run i due nuovi seed `43` e `44`. Per ogni seed si addestrano B e il finalista
con la recipe architetturale invariata; il seed 42 già disponibile completa una terna comune.

Si riportano per seed e aggregati:

- Macro-F1 e accuracy, media, deviazione standard e delta appaiato candidata−B;
- Acc1/Acc2, confusioni e numero di successi discordanti;
- accuracy/F1 PrefixAUC e punti a 1/1,5/2 s;
- epoca best, gap train-validation, clipping, overflow AMP, tempo e memoria di training;
- parametri, stato/traffico, MAC multivalore, AC/SOP potenziali e ad attività, firing e proxy
  Horowitz del best, sul medesimo campione di profiling.

Con tre seed non si presenta un p-value fra seed come prova definitiva. La conclusione usa segno e
ampiezza dei tre delta, media/deviazione e coerenza delle analisi appaiate entro seed. Se il
vantaggio F1 cambia segno in almeno due seed, la struttura non è congelata come superiore.

## Fase B — trasferimento strutturale

DVS-Gesture è il benchmark di trasferimento già implementato. Si conserva la topologia scelta;
si adattano soltanto head, finestra/binning e augmentation imposti dal protocollo del dataset. I
ritardi restano `[1,2,4,8]` in unità di bin, come nell'architettura congelata
**F+DWC-3+TCAP-d8**: non si esegue un tuning specifico per il dataset. Il flip orizzontale resta
disabilitato perché cambia le classi sinistra/destra.

Il primo confronto usa seed 42 contro la baseline DVS-Gesture esistente. Se il delta è positivo,
si completano i seed 43 e 44 per B e candidata; se è negativo, si registra la mancata
trasferibilità e non si adatta TCAP al validation set. Il risultato distingue un miglioramento
generale da un vantaggio specifico per la dinamica del labiale.

## Fase C — raffinamento supervisionato della struttura congelata

Le modifiche si provano in sequenza. Ogni stadio usa il vincitore dello stadio precedente e viene
scartato al primo delta negativo replicato. Nessuna griglia di iperparametri.

### C1. Temporal Maskout

È la prima priorità. [G2N2](https://proceedings.bmvc2023.org/660/) riporta su DVS-Lip guadagni
tra 4,8 e 7,4 pp cancellando intervalli temporali; la configurazione migliore usa otto maschere
lunghe fino a 200 ms. Il codice locale implementa già la stessa semantica sui frame.

La prova fissata è `temporal_mask_count=8`, `temporal_mask_max_steps=4`: otto intervalli casuali
da 50–200 ms, con possibili sovrapposizioni. Si confronta la stessa augmentation su B e candidata,
così il guadagno della recipe non viene attribuito alla struttura. È il test con evidenza più
diretta e il miglior rapporto beneficio/costo.

### C2. Augmentation geometrica neuromorfica

Solo se C1 è positivo si aggiunge una policy moderata e unica, senza ricerca automatica. La
[Neuromorphic Data Augmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670623.pdf)
mostra che flip, piccoli roll/rotazioni, cutout e shear applicati coerentemente a tutti i timestep
stabilizzano la generalizzazione su più benchmark DVS. Per DVS-Lip si mantiene il flip già usato e
si usa la policy di intensità più bassa del paper; trasformazioni che spostano la bocca fuori dal
campo vengono rifiutate nel test di integrità. CutMix viene lasciato a un'eventuale prova separata,
per non confondere subito invarianti geometriche e target misti.

### C3. Supervisione temporale tardiva

PLIF ha mostrato +3,47 pp di F1-PrefixAUC e circa +10 pp tra 100 e 300 ms dopo l'ultimo evento,
pur non migliorando il punto finale. MG+TCAP mostra anch'esso un vantaggio maggiore ai prefissi che
a 2 s. Questi risultati motivano una loss ausiliaria sui prefissi tardivi, non un readout
`last_event+K`.

Si mantiene la cross-entropy label-smoothed sul logit finale e si aggiunge la stessa loss ai
prefissi fissi 1,0 e 1,5 s, mediata una sola volta. Non si supervisionano i primi 500 ms, dove
l'informazione di parola è insufficiente. La prova è affine al principio di
[Temporal Efficient Training](https://openreview.net/pdf?id=_XNtisL32jv), che ottimizza le uscite
nel tempo per ottenere minimi più generalizzabili, ma viene dichiarata come variante late-prefix
specifica per sequenze dinamiche. Si promuove soltanto se non riduce il F1 finale e aumenta il
PrefixAUC in entrambi i seed di conferma.

### C4. Orizzonte di training e clipping

F+TCAP seleziona il best all'epoca 122/128 e MG all'epoca 127: il cosine termina mentre i modelli
stanno ancora migliorando. Inoltre il clipping globale a 1,0 interviene nel 99,998% degli step di
F+TCAP, con norma pre-clip mediana circa 42,7; gli overflow AMP sono soltanto 0,059%.

Si esegue prima un solo training a 192 epoche, cosine riscalato e sei epoche di warmup, mantenendo
il clipping. Solo se il gradiente resta finito e la curva continua a essere limitata dalla clip si
ammette un confronto senza clipping globale; non si provano dieci soglie. Weight decay, learning
rate di picco e surrogate non vengono toccati nello stesso run.

### C5. Media dei pesi e calibrazione

Se le ultime epoche oscillano attorno allo stesso plateau, si salva una piccola finestra di
checkpoint e si valuta una sola media dei pesi prima di avviare nuovi training. Temperature
scaling può calibrare la confidenza ma non è una tecnica per aumentare F1/accuracy e resta fuori
dalla selezione prestazionale.

## Fase D — JEPA-like e predictive coding

Questa fase avviene **dopo** conferma multi-seed, trasferimento e definizione di un forte riferimento
supervisionato, ma prima dell'unico accesso al test ufficiale. Il motivo è attributivo: pretraining e
fine-tuning devono essere confrontati con la stessa architettura e la stessa recipe già fissate.

Non esiste ancora evidenza diretta sufficiente per dichiarare JEPA la scelta migliore su DVS-Lip.
[Masked Event Modeling](https://openaccess.thecvf.com/content/WACV2024/papers/Klenk_Masked_Event_Modeling_Self-Supervised_Pretraining_for_Event_Cameras_WACV_2024_paper.pdf)
dimostra però che il masked pretraining su eventi può trasferire a task semantici; NDA mostra anche
un piccolo vantaggio del pretraining contrastivo non supervisionato rispetto al training da zero
nel proprio trasferimento CIFAR10-DVS.

Il primo e unico esperimento esplorativo consigliato è una predizione **latente** di blocchi
spaziotemporali mascherati/futuri con encoder target EMA e predictor leggero, seguita da fine-tuning
completo. Evita di ricostruire count rumorosi pixel per pixel e mantiene il backbone finale. Il
controllo è il miglior training supervisionato from-scratch con stessi seed, dati etichettati e
budget di fine-tuning. Si registra separatamente il costo di pretraining. Se non migliora la media
F1, non si aprono varianti di mask ratio, predictor o teacher.

## Fase E — compressione e valutazione finale

Dopo aver scelto fra training supervisionato e pretraining si valuta `TCAP → T` depthwise,
grouping/pruning dei mixer e quantization-aware training. Ogni variante deve riportare perdita di
F1 insieme a stato, traffico, firing e proxy energetiche. Questa è una frontiera Pareto separata:
non sostituisce il modello prestazionale nelle conclusioni di accuracy.

L'official test viene aperto soltanto quando sono fissati il modello prestazionale, l'eventuale
modello compresso, la recipe e la regola di checkpoint selection. Le configurazioni finali vengono
valutate in un'unica campagna, senza usare l'esito del test per scegliere fra varianti.

## Ordine operativo e stop rule

| Ordine | Esperimento | Full massimi prima dello stop |
|---:|---|---:|
| 0 | ablation checkpoint-only dei tap TCAP | 0 |
| 1 | high-frequency DWC-3; eventuale TCAP `[1,2,4,8]` solo se autorizzato | 1–2 |
| 2 | B e finalista, seed 43/44 | 4 |
| 3 | trasferimento DVS-Gesture, prima seed 42 | 1, poi 4 solo se positivo |
| 4 | Maskout su B e finalista | 2 totali sul seed di screening |
| 5 | augmentation geometrica, late-prefix loss, 192 epoche | 1 stadio alla volta |
| 6 | JEPA-like/predictive pretraining | 1 pretraining + fine-tuning controllato |
| 7 | compressione/quantizzazione e campagna official-test | dopo il modello prestazionale |

Si interrompe una famiglia quando il suo primo test preregistrato è negativo o quando per salvarla
servirebbe scegliere a posteriori fra più kernel, tau, ritardi, mask ratio o learning rate. Il test
ufficiale viene usato soltanto dopo aver fissato architettura, recipe e regola di selezione.
