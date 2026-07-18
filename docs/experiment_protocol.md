# Protocollo sperimentale

## Principio

Ogni esperimento deve rispondere a una sola domanda. Un run non è giustificato se non è chiaro quale ipotesi possa supportare o smentire.

## Procedura

1. eseguire i test automatici;
2. usare la configurazione smoke dopo modifiche strutturali;
3. lanciare un solo seed di sviluppo;
4. ispezionare validation, perturbazioni e profili;
5. replicare con almeno tre seed soltanto configurazioni decisive;
6. usare il test set una volta per seed, sul miglior checkpoint di validation;
7. documentare il commit Git e la configurazione risolta.

## Run ID

Il run ID include nome, timestamp e seed. Esempio:

```text
phase1_dvsgc_order2_miniqkformer__20260717_210000__seed42
```

## Logging

Vengono salvati:

- una riga per epoca in `history.csv`;
- messaggi essenziali in `run.log`;
- un JSON riepilogativo;
- una confusion matrix;
- un profilo aggregato;
- predizioni compresse soltanto nell'audit.

Non vengono salvate attivazioni complete o log per batch.

## Interpretazione delle perturbazioni

- `reverse_time`: distrugge ordine e direzione interna di ogni gesto;
- `shuffle_time`: distrugge la continuità mantenendo il contenuto dei frame;
- `reverse_segments`: approssima lo scambio dell'ordine delle primitive;
- `reverse_segments + reverse_class`: verifica se il modello riconosce la classe trasformata.

Nessuna perturbazione isolata dimostra comprensione causale. I risultati devono essere interpretati congiuntamente.
