# Roadmap decisiva

**Aggiornata:** 2026-09-13

## Esito della discovery

F+MG-Cap+TCAP stabilisce il record development a seed 42 con 53,15% Macro-F1, ma aggiunge soltanto
0,80 pp a F+TCAP e non raggiunge la soglia preregistrata di +2 pp. Il confronto appaiato attraversa
zero e il costo cresce nettamente. **F+TCAP resta quindi il finalista strutturale primario:** 52,36%
F1, 492.516 parametri e profilo migliore di B per stato, SOP e proxy Horowitz.

TCAP è il meccanismo più solido: produce +6,21 pp F1 su F e +4,73 pp su MG. MG resta evidenza
positiva sulla rappresentazione fine e sulla latenza, ma non entra nella campagna multi-seed
principale. PLIF, gated, TBR, Spike-TBR e clock matching sono chiusi come candidati di accuracy.

## Ultimo gate architetturale

Prima del freeze sono ammesse soltanto due verifiche:

1. ablation checkpoint-only dei tap TCAP 1/2/4; un run `[1,2,4,8]` viene autorizzato solo se il tap
   più lungo mostra un contributo marginale ancora forte;
2. un solo probe high-frequency DWC-3 nel primo stage di F+TCAP, con E0, stage 2 e recipe invariati.

La soglia di promozione resta +2 pp F1. Il probe locale può essere conservato come variante Pareto
se rimane entro −0,5 pp e riduce in modo misurato SOP/energia. Non si provano kernel, posizioni,
larghezze, tau, gain o fusioni alternative. Se entrambi i test passano si consente una sola
combinazione; altrimenti si congela il vincitore individuale o F+TCAP.

## Fase successiva

La roadmap dettagliata è in [`VALIDATION_REFINEMENT_ROADMAP.md`](VALIDATION_REFINEMENT_ROADMAP.md).
L'ordine vincolante è:

1. conferma su DVS-Lip con B e finalista ai seed 43 e 44;
2. trasferimento strutturale su DVS-Gesture, prima seed 42 e poi replica solo se positivo;
3. raffinamento supervisionato sequenziale: temporal Maskout, augmentation geometrica moderata,
   loss ai prefissi tardivi, orizzonte di training e media dei pesi;
4. un eventuale pretraining JEPA-like/predictive sul backbone ormai fissato;
5. compressione TCAP→T, pruning/grouping e quantization-aware training come fronte Pareto;
6. official test in un'unica campagna dopo il congelamento di modelli e recipe.

JEPA/predictive coding avviene dopo la conferma strutturale e la definizione del riferimento
supervisionato, ma prima dell'official test. In questo modo misura il valore del pretraining senza
confonderlo con una nuova architettura o con una recipe ancora mobile.

## Stop rule

Ogni famiglia riceve una configurazione fissata e viene chiusa al primo risultato negativo
replicato o quando richiederebbe scegliere a posteriori fra più valori. L'official test resta
embargoed. Tutti i confronti riportano F1/accuracy, Acc1/Acc2, PrefixAUC, costo di training e profilo
hardware v4 del proprio best.
