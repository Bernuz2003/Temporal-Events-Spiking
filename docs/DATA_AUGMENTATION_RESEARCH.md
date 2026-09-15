# Ricerca sulle data augmentation neuromorfiche

Stato del documento: **15 settembre 2026**.

Questo documento governa la selezione delle augmentation dopo il congelamento della struttura
**F+DWC-3+TCAP-d8**. Lo scopo è massimizzare il guadagno rispetto al modello congelato con pochi
run informativi, mantenendo separati tre concetti:

- **supporto in letteratura**: un risultato pubblicato, valutato nel proprio protocollo;
- **supporto nel repository**: una trasformazione implementata e configurabile;
- **evidenza locale**: un delta misurato a parità di modello, split, seed e training budget.

I valori pubblicati non sono confrontabili in assoluto con i nostri: cambiano split, numero di
frame, risoluzione, backbone e spesso anche l'uso del test set. Servono a scegliere quali ipotesi
provare per prime. Lo stato locale viene aggiornato solo quando il relativo artifact è completo.

## Legenda e regola sperimentale

| Simbolo | Stato | Significato |
|---|---|---|
| **✓** | completato | artifact disponibile e risultato interpretabile |
| **⏳** | in corso | run avviato, risultato non ancora disponibile |
| **○** | pianificato | candidato selezionato, non ancora implementato o lanciato |
| **—** | escluso | incompatibile con le classi o a ROI insufficiente |

Lo screening confronta ogni singola augmentation con il checkpoint architetturale congelato dello
stesso dataset. I candidati indipendenti possono essere eseguiti in parallelo. Si combina una
tecnica soltanto dopo un delta positivo isolato e si prova una sola combinazione forward, evitando
una ricerca sul prodotto cartesiano di trasformazioni e intensità. La validation non riceve
augmentation.

Su DVS-Lip il criterio fissato per un singolo run seed 42 è:

- almeno **+1,0 punto percentuale di Macro-F1**;
- perdita di accuracy non superiore a **0,5 punti**.

Se Temporal Maskout e spatial erasing superano entrambi la soglia, l'unica combinazione ammessa è
Maskout + erasing e deve migliorare il migliore singolo di almeno **0,5 punti di Macro-F1**. Solo
la policy finale viene replicata sui seed 43 e 44.

## Catalogo compatto delle augmentation rilevanti

La tabella non tenta di enumerare ogni perturbazione possibile. Comprende le famiglie plausibili
per event stream discretizzati, con evidenza sufficiente o con una chiara ipotesi di invarianza.

| Famiglia | Tecnica | Trasformazione e ipotesi | Supporto corrente | Priorità |
|---|---|---|---|---|
| Occlusione temporale | **Temporal Maskout** | Azzera intervalli temporali contigui; riduce la dipendenza da segmenti specifici e simula eventi mancanti | generica, train-only | alta DVS-Lip; media DVS-Gesture |
| Occlusione spaziale | **Spatial erasing / Cutout** | Azzera regioni coerentemente su tutti i timestep; forza l'uso distribuito dei motion cue | generica, train-only | alta DVS-Lip; parte di NDA su CIFAR10-DVS |
| Geometrica | **Flip orizzontale** | Riflette le coordinate; valido solo quando l'etichetta è invariata | generica, con veto DVS-Gesture | già attivo DVS-Lip; escluso DVS-Gesture |
| Geometrica | Roll/traslazione | Sposta l'intera sequenza con wrap o padding; simula variazioni di posizione | non implementata | alta dentro NDA su CIFAR10-DVS |
| Geometrica | Rotazione | Ruota tutti i timestep con gli stessi parametri | non implementata | alta dentro NDA su CIFAR10-DVS |
| Geometrica | Zoom/crop | Cambia scala mantenendo coerenza temporale | non implementata | secondaria DVS-Lip; parte di pipeline pubblicate |
| Geometrica | Shear | Applica una deformazione affine coerente nel tempo | non implementata | alta dentro NDA su CIFAR10-DVS |
| Temporale | Time scaling/warping | Rallenta o accelera l'evento o la sequenza senza conoscere la classe | non implementata | seconda scelta DVS-Lip |
| Temporale | Shift/crop temporale | Trasla o ritaglia la sequenza nella finestra osservata | non implementata | diagnostica; rischio di rimuovere parti discriminative |
| Event-space | **EventDrop** | Elimina eventi casuali, intervalli o aree direttamente sullo stream raw | non implementata | bassa: evidenza non diretta sui tre dataset |
| Event-space | Timestamp jitter / coordinate noise / polarity noise | Perturba il sensore simulando rumore e quantizzazione | non implementata | bassa finché non è caratterizzato il rumore reale |
| Sample mixing | **MixUp** | Combina due tensori e target con peso continuo | non implementata; richiede target misti nel trainer | media DVS-Gesture/CIFAR10-DVS |
| Sample mixing | **CutMix** | Sostituisce una regione spazio-temporale e miscela i target | non implementata; batch-level | media DVS-Gesture; parte di NDA |
| Sample mixing | **EventMix** | Usa maschere 3D spazio-temporali e assegna i target in base alla distribuzione degli eventi | generica, batch-level | alta DVS-Gesture e CIFAR10-DVS |
| Policy | **NDA-M1N2** | Flip e CutMix più una trasformazione geometrica campionata a intensità moderata | non implementata | alta CIFAR10-DVS |

Le trasformazioni fotometriche convenzionali non sono prioritarie: i nostri input sono conteggi di
eventi ON/OFF, non immagini RGB. Nel confronto della
[Neuromorphic Data Augmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670623.pdf)
su CIFAR10-DVS, le perturbazioni geometriche risultano nettamente più efficaci delle varianti
fotometriche trasferite dalle immagini.

## Matrice di priorità per dataset

| Tecnica | DVS-Lip | DVS-Gesture | CIFAR10-DVS | Motivazione principale |
|---|---:|---:|---:|---|
| Temporal Maskout | **P1** | P2 | P3 | forte ablation diretta su DVS-Lip; trasferimento sugli altri non ancora dimostrato |
| Spatial erasing | **P1** | P3 | dentro NDA | usato su DVS-Lip; componente geometrica consolidata, ma contributo isolato non pubblicato |
| EventMix | P3 | **P1** | **P1** | guadagni diretti e coerenti su DVS-Gesture e DVS-CIFAR10 |
| NDA-M1N2 | P3 | — | **P1** | ablation diretta su CIFAR10-DVS; flip semanticamente problematico su DVS-Gesture |
| Time scaling | P2 | P3 | P3 | plausibile per variazioni di velocità nel lip-reading, ma manca evidenza locale |
| EventDrop | P3 | P3 | P3 | metodo generale; pubblicato su N-Caltech101 e N-Cars, non sui tre benchmark qui considerati |

`P1` identifica il prossimo test ad alto ROI, `P2` un candidato subordinato a un risultato
positivo o a un collo di bottiglia ancora aperto, `P3` una riserva. L'assenza di priorità non è una
tesi di inefficacia generale: indica che non giustifica un run prima dei candidati meglio
supportati.

## DVS-Lip

### Evidenza esterna

L'evidenza più diretta riguarda **Temporal Maskout**. Nell'ablazione di
[G2N2, BMVC 2023](https://papers.bmvc2023.org/0660.pdf), lo stesso modello passa dal 58,7% al
66,1% di accuracy usando otto maschere lunghe da uno a sei intervalli da 33 ms: **+7,4 punti**.
Le maschere possono sovrapporsi. Il lavoro include anche flip orizzontale e distorsione della
velocità, ma non fornisce per questi un delta isolato equivalente. La selezione pubblicata usa un
protocollo diverso dal nostro e non rende trasferibile il guadagno numerico.

Il [codice ufficiale SpikGRU-DVSLip](https://github.com/manondampfhoffer/SpikGRU-DVSLip/blob/main/DVSLip.py)
usa inoltre una pipeline spaziale con quattro Cutout fino a 20 pixel e zoom. Questo rende lo
spatial erasing un candidato pertinente, ma **non prova il suo contributo individuale**, perché è
inserito in una ricetta più ampia con backbone e discretizzazione differenti.

La time masking è risultata centrale anche in uno studio sistematico sul lip-reading RGB
([Ma et al., 2022](https://arxiv.org/abs/2209.01383)); è un supporto indiretto alla robustezza
temporale, non evidenza neuromorfica aggiuntiva.

### Riferimento locale e stato

Il riferimento dello screening è F+DWC-3+TCAP-d8, seed 42. La sua ricetta contiene già
`horizontal_flip_probability=0.5`; perciò “senza nuova augmentation” significa **flip soltanto**,
non assenza assoluta di augmentation.

| Esperimento/policy | Stato | Accuracy | Macro-F1 | Delta attribuibile all'augmentation | Decisione |
|---|---:|---:|---:|---:|---|
| Frozen, flip 0,5 — seed 42 | **✓** | 55,53% | 55,18% | riferimento | comparatore C1/C2 |
| Frozen, flip 0,5 — seed 43 | **✓** | 55,09% | 54,88% | replica architetturale | non è uno screen augmentation |
| Frozen, flip 0,5 — seed 44 | **✓** | 56,73% | 56,56% | replica architetturale | non è uno screen augmentation |
| Media 3 seed | **✓** | **55,78 ± 0,85%** | **55,54 ± 0,90%** | riferimento di robustezza | deviazione standard campionaria |
| + Temporal Maskout `8 × [1,4]` bin | **⏳** | — | — | in attesa | confrontare col seed 42 |
| + Spatial erasing `4 × [1,20]` px | **⏳** | — | — | in attesa | confrontare col seed 42 |
| + Maskout + spatial erasing | **○** | — | — | non misurato | solo se entrambi i singoli passano |
| Time scaling | **○** | — | — | non misurato | P2, senza sweep di velocità |

Il flip non è stato isolato rispetto a una ricetta identica con probabilità zero; il suo delta
locale è quindi **non identificabile** dagli artifact esistenti. Non si spende un nuovo full run
solo per ricostruirlo: resta parte del riferimento congelato.

## DVS-Gesture

### Evidenza esterna e vincoli semantici

L'evidenza più forte è **EventMix**. Nello stesso protocollo di
[EventMix](https://floyedshen.github.io/pdf/shen2023eventmix.pdf), i guadagni su DVS-Gesture sono:

| Backbone pubblicato | Senza mixing | MixUp | CutMix | EventMix | Delta EventMix |
|---|---:|---:|---:|---:|---:|
| ResNet-34 | 86,33% | 89,06% | 87,11% | **91,80%** | **+5,47 pp** |
| ResNet-18 | 85,55% | — | — | **89,45%** | **+3,90 pp** |
| MobileNetV2 | 78,91% | — | — | **82,93%** | **+4,02 pp** |

La coerenza su tre backbone rende EventMix il candidato P1, pur senza rendere confrontabili i
valori assoluti con il nostro protocollo. EventMix non è un semplice CutMix: genera una maschera
spazio-temporale 3D e pesa il target usando la distribuzione degli eventi effettivamente mescolati.

Il flip orizzontale è escluso nel protocollo locale: DVS-Gesture contiene classi direzionali, tra
cui sinistra/destra e rotazioni orarie/antiorarie. Una riflessione senza rimappatura delle label
introduce rumore supervisionato. Una futura trasformazione con rimappatura esplicita sarebbe una
tecnica diversa e richiederebbe un test dedicato.

### Riferimento locale e stato

| Esperimento/policy | Stato | Accuracy | Macro-F1 | Effetto misurato | Decisione |
|---|---:|---:|---:|---:|---|
| Baseline B, nessuna augmentation — seed 42 | **✓** | 84,47% | 83,62% | riferimento architetturale | storico |
| Frozen, nessuna augmentation — seed 42 | **✓** | **89,39%** | **88,81%** | +4,92 pp Acc; +5,19 pp F1 vs B | riferimento per il raffinamento |
| EventMix sul frozen | **○** | — | — | config pronta | **screen P1** |
| Temporal Maskout `8 × [1,5]` bin | **○** | — | — | config pronta | screen parallelo P2 |
| MixUp/CutMix separati | **○** | — | — | non misurato | solo se servono a interpretare EventMix |
| Flip orizzontale | **—** | — | — | non valido | escluso senza label remapping |

Il +5,19 pp della struttura congelata non è un effetto di augmentation: definisce il nuovo
comparatore. Ogni futura policy DVS-Gesture va confrontata con **88,81% Macro-F1** sul seed 42, non
con B.

## CIFAR10-DVS

### Stato di integrazione

Il dataset non è ancora supportato nel repository. Non esistono quindi una baseline locale, un
trasferimento della struttura congelata o run di augmentation interpretabili. Prima dello
screening vanno congelati split, numero di timestep, risoluzione e reference config; non si sceglie
la policy osservando il test set.

### Evidenza esterna

Le due tecniche con il supporto diretto più forte sono **NDA** ed **EventMix**.

[NDA, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670623.pdf)
combina flip e CutMix con trasformazioni geometriche coerenti nel tempo. La configurazione M1N2
campiona una sola operazione fra roll, rotazione, cutout e shear a intensità moderata; è la migliore
della relativa ablation. Sul protocollo pubblicato:

| Confronto NDA su CIFAR10-DVS | Senza NDA | Con NDA-M1N2 | Delta |
|---|---:|---:|---:|
| ResNet-19 | 67,9% | **78,0%** | **+10,1 pp** |
| VGG-11 | 76,2% | **79,6%** | **+3,4 pp** |
| VGG-11 full-resolution | 76,3% | **81,7%** | **+5,4 pp** |

Nella stessa ablation principale, M1N2 raggiunge 78,0%, contro 73,4% di M1N1, 75,1% di M2N2 e
71,4% di M3N3. Questo giustifica una configurazione iniziale unica e vieta uno sweep locale su M e
N.

[EventMix](https://floyedshen.github.io/pdf/shen2023eventmix.pdf) fornisce un secondo risultato
diretto:

| Backbone pubblicato | Baseline | MixUp | CutMix | EventMix | Delta EventMix |
|---|---:|---:|---:|---:|---:|
| ResNet-34 | 81,13% | 82,93% | 83,15% | **85,60%** | **+4,47 pp** |
| ResNet-18 | 80,24% | — | — | **84,38%** | **+4,14 pp** |
| MobileNetV2 | 79,46% | — | — | **83,70%** | **+4,24 pp** |

I risultati NDA usano, tra le altre scelte, split 9:1, 10 frame e spesso input 48×48. Quelli
EventMix adottano ancora un altro setup. I due metodi vanno quindi confrontati localmente contro
la medesima reference; la tabella non stabilisce a priori quale vincerà sul nostro modello.

### Piano locale

| Esperimento/policy | Stato | Risultato locale | Decisione |
|---|---:|---:|---|
| Supporto dataset e reference frozen | **○** | assente | prerequisito |
| NDA-M1N2 | **○** | non misurato | screen P1, una configurazione fedele |
| EventMix | **○** | non misurato | screen P1 in parallelo a NDA |
| Migliore singolo multi-seed | **○** | non misurato | solo dopo lo screening |
| NDA + EventMix | **—** | non misurato | nessuna combinazione iniziale: confonderebbe due policy composte |
| Photometric/color policy | **—** | non misurato | evidenza pubblicata sfavorevole e semantica debole sui count |

NDA-M1N2 ed EventMix rispondono a ipotesi diverse e possono essere sottoposti a screening in
parallelo. Non si scompone preventivamente NDA in una griglia di roll, rotazione, cutout e shear;
la policy pubblicata è il candidato. Non si combina NDA con EventMix nella prima campagna.

## Stato dell'infrastruttura condivisa

La configurazione espone già un solo campo `augmentation`, indipendente dal dataset. Il wrapper
comune applica oggi:

```yaml
augmentation:
  horizontal_flip_probability: 0.0
  temporal_mask_count: 0
  temporal_mask_max_steps: 0
  spatial_erasing_count: 0
  spatial_erasing_max_pixels: 0
  event_mix_probability: 0.0
```

Queste trasformazioni sono implementate una volta e riusate dagli adapter DVS-Lip e DVS-Gesture.
Il workflow di raffinamento verifica che modello, rappresentazione, evaluation e training restino
allineati alla reference, identifica la famiglia modificata e distingue screening singolo e
combinazione. L'augmentation viene disabilitata in validation e nelle valutazioni finali.

EventMix è implementato una sola volta nel training engine: genera maschere GMM 3D, usa partner
senza self-mixing e combina le due cross-entropy per sample con il peso derivato dalla distanza fra
event stream mediati spazialmente. I parametri non completamente specificati dall'articolo
(intervallo delle scale GMM, griglia della maschera e pooling della distanza) sono salvati nella
config risolta.

Le estensioni future devono rispettare la stessa separazione:

- trasformazioni di un singolo sample, come geometria e masking, nel wrapper condiviso;
- trasformazioni fra sample nel componente batch-level comune del training engine; EventMix è già
  supportato, mentre MixUp e CutMix restano candidati;
- parametri e compatibilità semantica nei file YAML del dataset, senza classi o workflow duplicati.

## Fonti primarie

- [G2N2: temporal Maskout su DVS-Lip, BMVC 2023](https://papers.bmvc2023.org/0660.pdf)
- [SpikGRU-DVSLip: implementazione pubblica delle augmentation](https://github.com/manondampfhoffer/SpikGRU-DVSLip/blob/main/DVSLip.py)
- [Neuromorphic Data Augmentation, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670623.pdf)
- [EventMix, Information Sciences 2023](https://floyedshen.github.io/pdf/shen2023eventmix.pdf)
- [EventDrop, IJCAI 2021](https://www.ijcai.org/proceedings/2021/0097.pdf)
- [Training Strategies for Improved Lip-Reading, 2022](https://arxiv.org/abs/2209.01383)
