# Direzioni di investigazione

Questa pagina raccoglie ipotesi, non moduli già approvati.

## 1. Rappresentazione event-native

Domanda: quanta informazione temporale task-relevant viene eliminata dalla formazione di pochi frame?

Influenze concettuali:

- Fast Feature Field: rappresentazione degli eventi appresa tramite predizione del futuro;
- codifiche set-based e coordinate-based;
- time surfaces e statistiche temporali locali.

Decisione attuale: non implementare F³. Prima misurare la perdita con sonde più semplici.

## 2. Stato temporale locale

Domanda: il singolo stato LIF è sufficiente a distinguere storie temporali differenti?

Influenze concettuali:

- LMUFormer e memorie state-space strutturate;
- delayed synapses;
- tracce temporali multiscala;
- neuroni con dinamiche più ricche.

Decisione attuale: non effettuare uno sweep di moduli. Dopo gli audit verrà scelto un solo candidato sulla base di requisiti scientifici e implementativi.

## 3. Training predittivo latente

Domanda: uno stato addestrato a prevedere una rappresentazione futura conserva dinamiche più utili rispetto alla sola classificazione?

Influenza principale: JEPA.

Decisione attuale: rimandato finché non disponiamo di un frontend temporale che meriti di essere preaddestrato.

## 4. Computazione prediction-error-driven

Domanda: una previsione accurata può giustificare il riuso dello stato o il salto di elaborazioni ridondanti?

Influenze:

- predictive coding spiking;
- dynamic computation;
- token pruning e early recognition.

Decisione attuale: prima simulazione offline, nessun gating hardware finché non emerge un Pareto improvement.

## Cosa non costituisce da solo un contributo centrale

- QKFormer + LMU;
- QKFormer + Mamba;
- aggiunta isolata di delayed convolution;
- una loss JEPA senza audit temporale;
- ottimizzazione della sola attention;
- copia del frontend F³.
