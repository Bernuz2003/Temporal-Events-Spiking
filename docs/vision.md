# Visione scientifica

## Problema

Le pipeline correnti per event-based vision applicano spesso due compressioni temporali scarsamente motivate:

1. il flusso asincrono DVS viene convertito in pochi frame regolari;
2. l'integrazione della storia è delegata quasi interamente al potenziale di membrana LIF.

La prima decisione può eliminare micro-timing e ordine prima che la rete abbia la possibilità di apprendere cosa sia utile. La seconda può offrire una memoria troppo povera per descrivere traiettorie e fasi temporali complesse.

## Ipotesi di lavoro

### H1 — Perdita da binning

Rappresentazioni che preservano timing e ordine dovrebbero risultare superiori ai frame di conteggio quando il task richiede realmente la temporalità.

### H2 — Limite dell'integrazione

Anche quando la temporalità è presente nell'input, un frontend basato quasi esclusivamente su convoluzioni spaziali e LIF potrebbe non conservarla adeguatamente.

### H3 — Ridondanza prevedibile

Se uno stato temporale locale evolve in modo prevedibile, parte della computazione potrebbe essere ridotta senza perdere informazione rilevante.

H3 non viene implementata nella Fase 1. Prima dobbiamo dimostrare che il protocollo misura davvero H1 e H2.

## Principio architetturale aperto

La direzione generale è investigare se la tokenizzazione DVS debba diventare il readout di uno **stato temporale locale persistente**, anziché la proiezione di frame quasi indipendenti.

Questa è un'ipotesi di ricerca, non una scelta già fissata. Il progetto deve poter portare a un'architettura molto diversa da QKFormer.

## Ruolo della baseline

Mini-QKFormer viene usata perché:

- è già stata profilata;
- offre una gerarchia spaziale e blocchi di attenzione spiking;
- il patch embedding concentra la maggior parte di parametri e SOP;
- permette confronti con il lavoro precedente.

Non è il modello da preservare. Ogni modulo può essere sostituito qualora i dati lo giustifichino.
