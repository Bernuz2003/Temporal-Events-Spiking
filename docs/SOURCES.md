# Fonti e selezione delle famiglie architetturali

**Aggiornato:** 2026-09-07

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

## Triage PLIF/PMSN/PSN/LMU/Mamba/GRU

| Famiglia | Evidenza utile | Rischio nel nostro protocollo | Decisione e trigger |
|---|---|---|---|
| **PSN channel-wise** | evidenza diretta DVS-Lip; memoria temporale di ordine basso | il risultato pubblicato confonde neuron model, Conv3d iniziale, backbone e readout | **Già coperto da T**. Nessun run PSN separato finché T non mostra un segnale positivo |
| **PLIF** | [Fang et al. 2021](https://arxiv.org/abs/2007.05785): costante di tempo apprendibile e minore sensibilità all'inizializzazione su benchmark neuromorfici | può produrre un guadagno piccolo e diffuso senza risolvere l'embedding | **Primo fallback**, un solo run per-channel se F/T indicano memoria insufficiente ma training stabile |
| **PMSN** | confronto DVS-Lip favorevole in [Neuromorphic Sequential Arena](https://arxiv.org/abs/2505.22035) | modello pubblicato circa 9.5M, readout/dense head e protocollo diversi; dinamica parallelizzata meno naturale per streaming stateful | nessun run ora; rivalutare solo se T aiuta molto e serve una memoria temporale più lunga |
| **GRU / SpikGRU** | [SpikGRU2+](https://openaccess.thecvf.com/content/CVPR2024W/EVW/html/Dampfhoffer_Neuromorphic_Lip-Reading_With_Signed_Spiking_Gated_Recurrent_Units_CVPRW_2024_paper.html) mostra che la ricorrenza gated è forte su DVS-Lip | sistema bidirezionale da decine di milioni di parametri, 90 bin e augmentation forte; il nostro gated globale storico è invalido | prima recuperare il piccolo readout causale con init verificata; GRU compatta solo se quel gate conserva memoria e migliora l'overfit |
| **LMU** | [LMUFormer](https://arxiv.org/abs/2402.04882) mostra memoria compatta, training parallelo e inferenza streaming su task di sequenza e speech | nessuna evidenza diretta DVS-Lip; ordine e finestra di memoria aprirebbero nuove scelte e l'integrazione richiederebbe una nuova architettura | fuori dal budget corrente; considerare soltanto se emerge una dipendenza lunga che FIR/PLIF non catturano |
| **Mamba** | modelli state-space efficaci su sequenze; [TVTA 2026](https://arxiv.org/abs/2607.08236) usa un modulo Mamba su DVS-Lip | il risultato DVS-Lip usa Mamba bidirezionale, supervisione visemica e un sistema più ampio; costo e causalità cambiano | rinviato; non è un'ablazione minima del modello corrente |

Questa graduatoria evita sei training comparativi senza eliminare le idee: PSN è testato tramite T;
Il gated corretto viene rilanciato nella prima coppia con F; PLIF conserva un trigger condizionale.
PMSN, LMU, Mamba e GRU richiedono un'evidenza locale che renda
plausibile il loro costo prima di ricevere budget.

## Riferimenti DVS-Lip e protocollo

- [MSTP, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Tan_Event-Based_Lip-Reading_With_Multi-Scale_Spatio-Temporal_Features_CVPR_2022_paper.html):
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
