# Stato corrente

**Aggiornato:** 2026-09-13

**Fase:** chiusura controllata della discovery; F+TCAP è il finalista strutturale corrente.

## Evidenza decisiva

La baseline B raggiunge 44,81% accuracy e 44,15% Macro-F1. F+TCAP raggiunge 52,79/52,36% con
492.516 parametri: +7,98/+8,21 pp su B e +0,54 pp F1 sul controllo 2M, usando il 75% di parametri in
meno. Il profilo misura 575.488 elementi di stato, 2.915,70 M SOP, 1.499,48 M MAC e proxy Horowitz
7.157,55 µJ ad attività / 9.526,10 µJ densa.

F+MG-Cap+TCAP stabilisce il record seed 42 con 53,52/53,15%, ma MG aggiunge a F+TCAP soltanto
0,73/0,80 pp. Il bootstrap appaiato F1 dà IC95% `[−1,16; +2,71]` e McNemar `p=0,479`. Il costo sale
invece a 541.476 parametri, 772.096 elementi di stato, 3.670,67 M SOP, 1.845,51 M MAC, 8.764,68 µJ
ad attività e 19,17 h di training. Non raggiunge la soglia preregistrata e resta record esplorativo,
mentre F+TCAP resta il finalista Pareto.

TCAP è il contributo robusto: +6,21 pp F1 quando viene aggiunto a F e +4,73 pp quando viene aggiunto
a MG. MG conserva un risultato di latenza: nel combinato migliora F+TCAP di 5,05 pp a 1 s e di 2,34
pp nel F1-PrefixAUC, ma il vantaggio si restringe a 0,80 pp a 2 s.

## PLIF e dinamica temporale

PLIF termina 0,47 pp F1 sotto B, ma migliora il F1-PrefixAUC di 3,47 pp e mantiene circa +10 pp tra
100 e 300 ms dopo l'ultimo evento. I tau appresi separano trasformazioni rapide e memoria profonda;
nessun cutoff event-aligned supera però il riferimento finale. PLIF non viene promosso come neuron
model. La sua evidenza motiva supervisione ai prefissi tardivi e, in seguito, un possibile arresto
adattivo basato sulla confidenza.

## Lavoro autorizzato prima del freeze

Restano soltanto:

- un solo run F+TCAP `[1,2,4,8]`, autorizzato dalla diagnostica: rimuovere `d=4` costa 51,60 pp F1
  contro 49,79 pp per `d=2`;
- config `dvslip_f_tcap_stage1_dwc3.yaml` per il probe F+TCAP con mixer locale depthwise 3×3 nel
  primo stage, motivato dal collo residuo su Acc1 e dalla letteratura high-frequency.

La promozione richiede +2 pp F1. Non si riaprono MG, TBR, Spike-TBR, PLIF, gated readout, altri
neuron model o rappresentazioni.

## Fase successiva

Dopo il gate si eseguono due seed nuovi comuni per B e finalista, trasferimento su DVS-Gesture e
raffinamento supervisionato sequenziale. JEPA-like/predictive pretraining viene valutato soltanto
sul backbone confermato e dopo una forte recipe supervisionata, prima dell'official test. Piano e
stop rule sono in [`VALIDATION_REFINEMENT_ROADMAP.md`](VALIDATION_REFINEMENT_ROADMAP.md).
